
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    AutoConfig,
    set_seed,
)
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import torch
import random
import os

set_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME        = "google-bert/bert-base-cased"
LEARNING_RATE     = 2e-5
NUM_EPOCHS        = 3
BATCH_SIZE        = 8
MAX_LENGTH        = 128
DATA_DIR          = "data_iob2"
SAVE_PATH         = "bert_ewt_ner"

# ── Label set ──────────────────────────────────────────────────────────────────
# EWT (Universal NER) uses only PER, ORG, LOC — no MISC.
# This is the label set the model is trained on and scores are reported against.
LABEL_LIST = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
label2id   = {l: i for i, l in enumerate(LABEL_LIST)}
id2label   = {i: l for i, l in enumerate(LABEL_LIST)}


# ── 1. Parse .iob2 files ───────────────────────────────────────────────────────

def parse_iob2(filepath, token_col=0, tag_col=1):
    """Read a .iob2 file into a list of {'tokens':..., 'ner_tags':...} dicts."""
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#") or line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) > max(token_col, tag_col):
                    tokens.append(parts[token_col])
                    tags.append(parts[tag_col])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def load_all_datasets(data_dir):
    """Load all three datasets from local .iob2 files."""
    ewt = {
        "train": parse_iob2(f"{data_dir}/universal_train.iob2", token_col=1, tag_col=2),
        "dev":   parse_iob2(f"{data_dir}/universal_dev.iob2",   token_col=1, tag_col=2),
        "test":  parse_iob2(f"{data_dir}/universal_test_masked.iob2", token_col=1, tag_col=2),
    }
    conll = {
        "train": parse_iob2(f"{data_dir}/news_train.iob2"),
        "dev":   parse_iob2(f"{data_dir}/news_dev.iob2"),
        "test":  parse_iob2(f"{data_dir}/news_test.iob2"),
    }
    wiesp = {
        "train": parse_iob2(f"{data_dir}/astro_train.iob2"),
        "dev":   parse_iob2(f"{data_dir}/astro_dev.iob2"),
        "test":  parse_iob2(f"{data_dir}/astro_test.iob2"),
    }
    return ewt, conll, wiesp


# ── 2. Convert to HuggingFace Dataset and tokenize ────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def to_hf_dataset(split_data):
    """Convert list of dicts to a HuggingFace Dataset."""
    return Dataset.from_dict({
        "tokens":   [ex["tokens"]   for ex in split_data],
        "ner_tags": [ex["ner_tags"] for ex in split_data],
    })


def normalize_tag(tag):
    """
    Map a tag to the EWT label set.
    - Tags already in LABEL_LIST pass through unchanged.
    - MISC (CoNLL) → O  (EWT has no MISC class)
    - WIESP domain tags → coarse PER/ORG/LOC or O
    - Anything else → O
    """
    if tag in label2id:
        return tag

    # CoNLL MISC → O
    if "MISC" in tag:
        return "O"

    # WIESP coarse mapping
    wiesp_map = {
        "Person":               ("B-PER", "I-PER"),
        "Facility":             ("B-ORG", "I-ORG"),
        "Observatory":          ("B-ORG", "I-ORG"),
        "Mission":              ("B-ORG", "I-ORG"),
        "Telescope":            ("B-ORG", "I-ORG"),
        "Instrument":           ("B-ORG", "I-ORG"),
        "CelestialObject":      ("B-LOC", "I-LOC"),
        "CelestialObjectRegion":("B-LOC", "I-LOC"),
        "Region":               ("B-LOC", "I-LOC"),
    }
    for entity_type, (b_tag, i_tag) in wiesp_map.items():
        if entity_type in tag:
            return b_tag if tag.startswith("B-") else i_tag

    return "O"


def tokenize_and_align_labels(examples):
    """
    Tokenize words and align NER labels to subword tokens.
    Subword continuations get label -100 (ignored in loss and evaluation).
    Tags are also normalized to the EWT label set.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=MAX_LENGTH,
        padding=False,
        truncation=True,
        is_split_into_words=True,
    )

    all_labels = []
    for batch_index, tags in enumerate(examples["ner_tags"]):
        word_ids     = tokenized_inputs.word_ids(batch_index=batch_index)
        label_ids    = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)                      # special token
            elif word_id == prev_word_id:
                label_ids.append(-100)                      # subword continuation
            else:
                normalized = normalize_tag(tags[word_id])
                label_ids.append(label2id[normalized])      # first subword of word

            prev_word_id = word_id

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def prepare_dataset(split_data):
    """Full pipeline: list of dicts → tokenized HuggingFace Dataset."""
    hf_ds = to_hf_dataset(split_data)
    return hf_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )


# ── 3. Evaluation function ────────────────────────────────────────────────────

metric = evaluate.load("seqeval")


def get_labels(predictions, references):
    """Convert integer predictions/references → string label lists, skip -100."""
    true_preds, true_refs = [], []
    for preds_row, refs_row in zip(predictions, references):
        pred_labels = []
        ref_labels  = []
        for p, r in zip(preds_row, refs_row):
            if r != -100:
                pred_labels.append(id2label[p])
                ref_labels.append(id2label[r])
        true_preds.append(pred_labels)
        true_refs.append(ref_labels)
    return true_preds, true_refs


def evaluate_split(model, dataloader, device, split_name=""):
    """Run inference on a dataloader and return seqeval metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
            batch       = {k: v.to(device) for k, v in batch.items()}
            outputs     = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            preds, labs = get_labels(predictions.cpu().tolist(),
                                     batch["labels"].cpu().tolist())
            all_preds.extend(preds)
            all_labels.extend(labs)

    results = metric.compute(predictions=all_preds, references=all_labels)
    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
    }


# ── 4. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load and prepare datasets ─────────────────────────────────────────────
    print("\nLoading datasets...")
    ewt, conll, wiesp = load_all_datasets(DATA_DIR)

    print("Tokenizing...")
    ewt_train_ds   = prepare_dataset(ewt["train"])
    ewt_dev_ds     = prepare_dataset(ewt["dev"])

    conll_dev_ds   = prepare_dataset(conll["dev"])
    conll_test_ds  = prepare_dataset(conll["test"])

    wiesp_dev_ds   = prepare_dataset(wiesp["dev"])
    wiesp_test_ds  = prepare_dataset(wiesp["test"])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_loader      = DataLoader(ewt_train_ds,  shuffle=True,  collate_fn=data_collator, batch_size=BATCH_SIZE)
    ewt_dev_loader    = DataLoader(ewt_dev_ds,    shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE)
    conll_dev_loader  = DataLoader(conll_dev_ds,  shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE)
    conll_test_loader = DataLoader(conll_test_ds, shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE)
    wiesp_dev_loader  = DataLoader(wiesp_dev_ds,  shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE)
    wiesp_test_loader = DataLoader(wiesp_test_ds, shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining on EWT...")
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Evaluate on EWT dev after each epoch
        ewt_dev_metrics = evaluate_split(model, ewt_dev_loader, device, "EWT dev")
        print(f"\nEpoch {epoch+1} | avg loss: {avg_loss:.4f} | EWT dev F1: {ewt_dev_metrics['f1']}")

        # Save best model based on EWT dev F1
        if ewt_dev_metrics["f1"] > best_f1:
            best_f1 = ewt_dev_metrics["f1"]
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  → New best model saved (F1={best_f1})")

    # ── Final cross-domain evaluation ─────────────────────────────────────────
    print("\nLoading best model for final evaluation...")
    model = AutoModelForTokenClassification.from_pretrained(SAVE_PATH)
    model.to(device)

    print("\n" + "="*55)
    print(f"{'Split':<25} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("="*55)

    for name, loader in [
        ("EWT dev (in-domain)",          ewt_dev_loader),
        ("CoNLL-2003 dev (similar)",      conll_dev_loader),
        ("CoNLL-2003 test (similar)",     conll_test_loader),
        ("WIESP-2022 dev (different)",    wiesp_dev_loader),
        ("WIESP-2022 test (different)",   wiesp_test_loader),
    ]:
        m = evaluate_split(model, loader, device, name)
        print(f"{name:<25} {m['f1']:>8} {m['precision']:>10} {m['recall']:>8}")

    print("="*55)
"""
data.py
-------
Handles everything data-related:
  - Parsing .iob2 files (EWT 5-col and CoNLL/WIESP 2-col)
  - Normalising tags from CoNLL/WIESP to the EWT label set
  - Tokenising and aligning labels to BERT subwords
  - Building HuggingFace Datasets and PyTorch DataLoaders
  - Saving EWT test predictions back to .iob2 format
"""

import os
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from config import LABEL_LIST, LABEL2ID, ID2LABEL


# ── WIESP → EWT coarse label mapping ─────────────────────────────────────────
# Maps WIESP entity type names to the closest EWT label.
# Proposal and URL have no equivalent → O.

# WIESP_COARSE = {
#     "Person":                "PER",
#     "Facility":              "ORG",
#     "Observatory":           "ORG",
#     "Mission":               "ORG",
#     "Telescope":             "ORG",
#     "Instrument":            "ORG",
#     "CelestialObject":       "LOC",
#     "CelestialObjectRegion": "LOC",
#     "Region":                "LOC",
# }


def normalize_tag(tag: str) -> str:
    """
    Map any NER tag to the 7-label EWT label set.

    Rules (in order):
      1. Already a valid EWT label → keep as-is
      2. Contains 'MISC' (CoNLL) → O
      3. Matches a WIESP entity type → coarse B-/I- PER/ORG/LOC
      4. Anything else → O
    """
    if tag in LABEL2ID:
        return tag

    if "MISC" in tag:
        return "O"

    for entity_type, coarse in WIESP_COARSE.items():
        if entity_type in tag:
            prefix = "B-" if tag.startswith("B-") else "I-"
            return f"{prefix}{coarse}"

    return "O"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_iob2(filepath: str, token_col: int = 0, tag_col: int = 1) -> List[Dict]:
    """
    Parse a .iob2 file into a list of sentence dicts.

    Args:
        filepath:  path to the .iob2 file
        token_col: column index for the token  (EWT = 1, CoNLL/WIESP = 0)
        tag_col:   column index for the NER tag (EWT = 2, CoNLL/WIESP = 1)

    Returns list of dicts:
        {
            "tokens":   ["The", "authors", ...],
            "ner_tags": ["O", "O", ...],
            "raw_lines": ["1\tThe\tO\t-\t-", ...],   # original lines, for prediction output
        }
    """
    sentences = []
    tokens, tags, raw_lines = [], [], []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip() or line.startswith("#"):
                if tokens:
                    sentences.append({
                        "tokens":    tokens,
                        "ner_tags":  tags,
                        "raw_lines": raw_lines,
                    })
                    tokens, tags, raw_lines = [], [], []
                continue

            parts = line.split("\t")
            if len(parts) > max(token_col, tag_col):
                tokens.append(parts[token_col])
                tags.append(parts[tag_col])
                raw_lines.append(line)

    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags, "raw_lines": raw_lines})

    return sentences


def load_all_datasets(data_dir: str) -> Dict[str, Dict]:
    """
    Load all three datasets from local .iob2 files.
    EWT uses token_col=1, tag_col=2 (5-column format).
    CoNLL and WIESP use token_col=0, tag_col=1 (2-column format).

    Returns:
        {
            "ewt":   {"train": [...], "dev": [...], "test": [...]},
            "conll": {"train": [...], "dev": [...], "test": [...]},
            "wiesp": {"train": [...], "dev": [...], "test": [...]},
        }
    """
    ewt_kwargs   = dict(token_col=1, tag_col=2)
    other_kwargs = dict(token_col=0, tag_col=1)

    datasets = {
        "ewt": {
            "train": parse_iob2(f"{data_dir}/universal_train.iob2",        **ewt_kwargs),
            "dev":   parse_iob2(f"{data_dir}/universal_dev.iob2",          **ewt_kwargs),
            "test":  parse_iob2(f"{data_dir}/universal_test_masked.iob2",  **ewt_kwargs),
        },
        "conll": {
            "train": parse_iob2(f"{data_dir}/news_train.iob2",  **other_kwargs),
            "dev":   parse_iob2(f"{data_dir}/news_dev.iob2",    **other_kwargs),
            "test":  parse_iob2(f"{data_dir}/news_test.iob2",   **other_kwargs),
        },
        "wiesp": {
            "train": parse_iob2(f"{data_dir}/astro_train.iob2", **other_kwargs),
            "dev":   parse_iob2(f"{data_dir}/astro_dev.iob2",   **other_kwargs),
            "test":  parse_iob2(f"{data_dir}/astro_test.iob2",  **other_kwargs),
        },
    }

    for name, splits in datasets.items():
        print(f"  {name:<6} — train: {len(splits['train'])}, "
              f"dev: {len(splits['dev'])}, test: {len(splits['test'])}")

    return datasets


# ── Tokenisation & label alignment ───────────────────────────────────────────

def make_tokenize_fn(tokenizer, max_length: int):
    """
    Returns a batched map function that:
      1. Tokenises words with BERT's subword tokeniser
      2. Aligns NER labels to subwords (-100 for continuations / special tokens)
      3. Normalises all tags to the EWT label set (not now but we may go back to it later)
    """
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            max_length=max_length,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        all_labels = []
        for batch_idx, tags in enumerate(examples["ner_tags"]):
            word_ids     = tokenized.word_ids(batch_index=batch_idx)
            label_ids    = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)                          # [CLS] / [SEP] / padding
                elif word_id == prev_word_id:
                    label_ids.append(-100)                          # subword continuation
                else:
                    #normalized = normalize_tag(tags[word_id])
                    #label_ids.append(LABEL2ID[normalized])          # first subword of word
                    label_ids.append(LABEL2ID.get(tags[word_id], LABEL2ID["O"]))

                prev_word_id = word_id

            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels


def prepare_split(sentences: List[Dict], tokenizer, max_length: int) -> Dataset:
    """Convert a list of sentence dicts into a tokenised HuggingFace Dataset."""
    hf_ds = Dataset.from_dict({
        "tokens":   [ex["tokens"]   for ex in sentences],
        "ner_tags": [ex["ner_tags"] for ex in sentences],
    })
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    return hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenising",
    )


def make_dataloader(dataset: Dataset, tokenizer, batch_size: int,
                    shuffle: bool = False) -> DataLoader:
    collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collator)


# ── Prediction output ─────────────────────────────────────────────────────────

def save_predictions(sentences: List[Dict], predictions: List[List[str]],
                     output_path: str):
    """
    Write model predictions to a .iob2 file in the same format as the input,
    replacing the NER tag column with the predicted tag.
    Compatible with the assignment's span_f1.py evaluation script.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sent, pred_tags in zip(sentences, predictions):
            for raw_line, pred_tag in zip(sent["raw_lines"], pred_tags):
                parts = raw_line.split("\t")
                parts[2] = pred_tag          # replace gold tag with prediction
                f.write("\t".join(parts) + "\n")
            f.write("\n")

    print(f"  Predictions saved → {output_path}")

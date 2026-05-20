"""
data.py — IOB2 parsing, tag normalisation, tokenisation, and DataLoader construction.

Exports: parse_iob2, normalize_tag, entity_density, load_all_datasets,
make_tokenize_fn, prepare_split, make_dataloader, save_predictions.
"""

import os
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from config import LABEL_LIST, LABEL2ID, ID2LABEL


UNIFY_MAP = {
    # WIESP labels that overlap with EWT/CoNLL concepts
    "B-Person":       "B-PER",
    "I-Person":       "I-PER",
    "B-Organization": "B-ORG",
    "I-Organization": "I-ORG",
    "B-Location":     "B-LOC",
    "I-Location":     "I-LOC",
}


def normalize_tag(tag: str) -> str:
    """
    Normalize a NER tag to the unified label set.
    Rules (in order):
      1. Unify overlapping labels (WIESP Person/Organization/Location --> PER/ORG/LOC)
      2. Map MISC --> O (CoNLL-specific for mixed entities, outside our scope)
      3. If the tag is in the label set --> keep as-is
      4. Anything else --> O
    """
    tag = UNIFY_MAP.get(tag, tag)
    if "MISC" in tag:
        return "O"
    return tag if tag in LABEL2ID else "O"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_iob2(filepath: str, token_col: int = 0, tag_col: int = 1,
               unit: str = "sentence") -> List[Dict]:
    """
    Parse a .iob2 file into a list of example dicts.

    Args:
        filepath:  path to the .iob2 file
        token_col: column index for the token  (EWT = 1, CoNLL/WIESP = 0)
        tag_col:   column index for the NER tag (EWT = 2, CoNLL/WIESP = 1)
        unit:      "sentence" (default) or "paragraph". Both modes parse blank-line
                   delimited blocks as one example — astro .iob2 files already store
                   one paragraph per block, so the parsing is identical and the
                   parameter is propagated as a semantic label only. Brief requires
                   no sentence-splitting and no merging of paragraphs.

    Returns list of dicts:
        {
            "id":        "<file_stem>_<idx:05d>",
            "tokens":    ["The", "authors", ...],
            "ner_tags":  ["O", "O", ...],
            "raw_lines": ["1\tThe\tO\t-\t-", ...],   # original lines, for prediction output
            "unit":      "sentence" | "paragraph",
        }
    """
    if unit not in ("sentence", "paragraph"):
        raise ValueError(f"unit must be 'sentence' or 'paragraph', got {unit!r}")

    stem = os.path.splitext(os.path.basename(filepath))[0]
    sentences: List[Dict] = []
    tokens, tags, raw_lines = [], [], []

    def _flush():
        if tokens:
            sentences.append({
                "id":        f"{stem}_{len(sentences):05d}",
                "tokens":    list(tokens),
                "ner_tags":  list(tags),
                "raw_lines": list(raw_lines),
                "unit":      unit,
            })
            tokens.clear(); tags.clear(); raw_lines.clear()

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip() or line.startswith("#"):
                _flush()
                continue

            parts = line.split("\t")
            if len(parts) > max(token_col, tag_col):
                tokens.append(parts[token_col])
                tags.append(parts[tag_col])
                raw_lines.append(line)

    _flush()
    return sentences


def entity_density(examples: List[Dict]) -> float:
    """
    Fraction of tokens carrying a non-"O" tag (after normalisation).

    Numerator: count of tokens whose normalised tag != "O".
    Denominator: total token count.
    Returns 0.0 if examples is empty.
    """
    n_total = 0
    n_entity = 0
    for ex in examples:
        for t in ex["ner_tags"]:
            n_total += 1
            if normalize_tag(t) != "O":
                n_entity += 1
    return (n_entity / n_total) if n_total else 0.0


def load_all_datasets(data_dir: str,
                      unit_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
    """
    Load all three datasets from local .iob2 files.
    EWT uses token_col=1, tag_col=2 (5-column format).
    CoNLL and WIESP use token_col=0, tag_col=1 (2-column format).

    Args:
        data_dir:        folder with .iob2 files.
        unit_overrides:  optional {dataset_key: unit} mapping to switch a dataset
                         from "sentence" (default) to "paragraph". Affects the
                         `unit` field on each example dict; parsing is the same.

    Returns:
        {
            "ewt":   {"train": [...], "dev": [...], "test": [...]},
            "conll": {"train": [...], "dev": [...], "test": [...]},
            "wiesp": {"train": [...], "dev": [...], "test": [...]},
        }
    """
    overrides = unit_overrides or {}
    ewt_unit   = overrides.get("ewt",   "sentence")
    conll_unit = overrides.get("conll", "sentence")
    wiesp_unit = overrides.get("wiesp", "sentence")

    ewt_kwargs   = dict(token_col=1, tag_col=2)
    other_kwargs = dict(token_col=0, tag_col=1)

    datasets = {
        "ewt": {
            "train": parse_iob2(f"{data_dir}/universal_train.iob2",        unit=ewt_unit, **ewt_kwargs),
            "dev":   parse_iob2(f"{data_dir}/universal_dev.iob2",          unit=ewt_unit, **ewt_kwargs),
            "test":  parse_iob2(f"{data_dir}/universal_test_masked.iob2",  unit=ewt_unit, **ewt_kwargs),
        },
        "conll": {
            "train": parse_iob2(f"{data_dir}/news_train.iob2",  unit=conll_unit, **other_kwargs),
            "dev":   parse_iob2(f"{data_dir}/news_dev.iob2",    unit=conll_unit, **other_kwargs),
            "test":  parse_iob2(f"{data_dir}/news_test.iob2",   unit=conll_unit, **other_kwargs),
        },
        "wiesp": {
            "train": parse_iob2(f"{data_dir}/astro_train.iob2", unit=wiesp_unit, **other_kwargs),
            "dev":   parse_iob2(f"{data_dir}/astro_dev.iob2",   unit=wiesp_unit, **other_kwargs),
            "test":  parse_iob2(f"{data_dir}/astro_test.iob2",  unit=wiesp_unit, **other_kwargs),
        },
    }

    for name, splits in datasets.items():
        print(f"  {name:<6} — train: {len(splits['train'])}, "
              f"dev: {len(splits['dev'])}, test: {len(splits['test'])}")

    return datasets


# ── Tokenisation & label alignment ───────────────────────────────────────────

def make_tokenize_fn(tokenizer, max_length: int, trunc_counter: Optional[List[int]] = None):
    """
    Returns a batched map function that:
      1. Tokenises words with BERT's subword tokeniser
      2. Aligns NER labels to subwords (-100 for continuations / special tokens)
      3. Normalises all the tags to the unified label set

    If `trunc_counter` (single-element list, mutable accumulator) is provided,
    increments it by the number of examples whose word-level coverage was cut
    off by truncation.
    """
    def tokenize_and_align_labels(examples):
        """
        Tokenise a batch of pre-tokenised sentences and align NER labels to subwords.

        Args:
            examples: batch dict with keys "tokens" (List[List[str]]) and
                      "ner_tags" (List[List[str]]).
        Returns:
            tokenized dict extended with "labels" (List[List[int]]) where -100
            marks special tokens and subword continuations, and valid positions
            hold integer label ids from LABEL2ID.
        """
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
                    normalized = normalize_tag(tags[word_id])
                    label_ids.append(LABEL2ID[normalized])          # first subword of word
                prev_word_id = word_id

            all_labels.append(label_ids)

            if trunc_counter is not None:
                covered = {w for w in word_ids if w is not None}
                if len(covered) < len(examples["tokens"][batch_idx]):
                    trunc_counter[0] += 1

        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels


def prepare_split(sentences: List[Dict], tokenizer, max_length: int,
                  return_truncation_count: bool = False):
    """
    Convert a list of example dicts into a tokenised HuggingFace Dataset.

    If return_truncation_count=True, returns (Dataset, n_truncated). Otherwise
    returns the Dataset alone (existing behaviour).
    """
    hf_ds = Dataset.from_dict({
        "tokens":   [ex["tokens"]   for ex in sentences],
        "ner_tags": [ex["ner_tags"] for ex in sentences],
    })
    trunc_counter = [0] if return_truncation_count else None
    tokenize_fn   = make_tokenize_fn(tokenizer, max_length, trunc_counter=trunc_counter)
    ds = hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenising",
        load_from_cache_file=False,
    )
    if return_truncation_count:
        return ds, trunc_counter[0]
    return ds


def make_dataloader(dataset: Dataset, tokenizer, batch_size: int,
                    shuffle: bool = False) -> DataLoader:
    """
    Wrap a HuggingFace Dataset in a DataLoader with token-classification padding.

    Args:
        dataset:    tokenised HuggingFace Dataset (output of prepare_split).
        tokenizer:  tokenizer used to determine pad token id.
        batch_size: number of examples per batch.
        shuffle:    whether to shuffle before each epoch (True for training).
    Returns:
        DataLoader that pads each batch to the longest sequence in that batch.
    """
    collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collator)


# ── Prediction output ─────────────────────────────────────────────────────────

def save_predictions(sentences: List[Dict], predictions: List[List[str]],
                     output_path: str):
    """
    Write model predictions to a .iob2 file,
    replacing the NER tag column with the predicted tag.
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

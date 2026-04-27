"""Subword label alignment + truncation accounting.

Same scheme as the baseline (data.py:128) — first subword gets the BIO
tag, continuation pieces get -100. Adds:
  • truncation counter so we can warn when too many paragraphs lose tail
    tokens at max_seq_len
  • returns a HuggingFace Dataset keyed by example id (so per-example
    metrics can be joined back).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from datasets import Dataset

import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import LABEL2ID  # noqa: E402
from data import normalize_tag  # noqa: E402  (root-level data.py, not src/data.py)
from src.data import Example  # noqa: E402


@dataclass
class TokenizationResult:
    dataset: Dataset
    n_truncated: int
    n_total: int


def tokenize_examples(examples: Sequence[Example], tokenizer,
                      max_seq_len: int) -> TokenizationResult:
    ids = [ex.id for ex in examples]
    token_lists = [ex.tokens for ex in examples]
    tag_lists = [ex.ner_tags for ex in examples]

    enc = tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )

    aligned_labels: List[List[int]] = []
    n_truncated = 0
    for i, tags in enumerate(tag_lists):
        word_ids = enc.word_ids(batch_index=i)
        labels: List[int] = []
        prev = None
        last_word_seen = -1
        for w in word_ids:
            if w is None:
                labels.append(-100)
            elif w == prev:
                labels.append(-100)
            else:
                labels.append(LABEL2ID[normalize_tag(tags[w])])
                last_word_seen = w
            prev = w
        aligned_labels.append(labels)
        if last_word_seen + 1 < len(token_lists[i]):
            n_truncated += 1

    ds = Dataset.from_dict({
        "id": ids,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": aligned_labels,
    })
    return TokenizationResult(dataset=ds, n_truncated=n_truncated, n_total=len(examples))

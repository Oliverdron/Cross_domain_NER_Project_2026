"""IOB2 loaders, hashing, deterministic injection pool.

Reuses the baseline's parse_iob2 logic. Paragraph-split files (e.g. astro)
are returned one example per paragraph; sentence-split files one per
sentence. The two cases are identical at the parser level — the spec
("paragraph" vs "sentence") only determines the unit count for budget
equivalence elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Example:
    id: str
    tokens: List[str]
    ner_tags: List[str]


def parse_iob2(filepath: str, token_col: int = 0, tag_col: int = 1,
               id_prefix: str = "") -> List[Example]:
    """Parse a .iob2 file into a list of Example objects.

    Empty lines and lines starting with '#' are sentence/paragraph
    separators. Every example gets a stable id of '<id_prefix>_<idx>'.
    """
    examples: List[Example] = []
    tokens: List[str] = []
    tags: List[str] = []

    def flush():
        if tokens:
            examples.append(Example(
                id=f"{id_prefix}_{len(examples)}",
                tokens=list(tokens),
                ner_tags=list(tags),
            ))
            tokens.clear()
            tags.clear()

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                flush()
                continue
            parts = line.split("\t")
            if len(parts) > max(token_col, tag_col):
                tokens.append(parts[token_col])
                tags.append(parts[tag_col])
    flush()
    return examples


def hash_file(filepath: str) -> str:
    """sha256 of the raw bytes of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split(path: str, name: str) -> List[Example]:
    """Auto-detect 2-col vs 5-col format and parse.

    EWT (universal_*) is 5-column with token at col 1, tag at col 2.
    CoNLL (news_*) and astro/WIESP are 2-column with token at col 0,
    tag at col 1.
    """
    base = os.path.basename(path).lower()
    if base.startswith("universal"):
        token_col, tag_col = 1, 2
    else:
        token_col, tag_col = 0, 1
    return parse_iob2(path, token_col=token_col, tag_col=tag_col, id_prefix=name)


# ── Injection pool ────────────────────────────────────────────────────────────

def build_injection_pool(examples: List[Example], seed: int) -> List[str]:
    """Deterministic ordering of target example ids. Same seed → same order.

    Returns a list of example ids in the order they should be injected.
    """
    rng = random.Random(seed)
    ids = [ex.id for ex in examples]
    rng.shuffle(ids)
    return ids


def persist_pool(pool: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for eid in pool:
            f.write(eid + "\n")


def load_pool(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def select_injected(pool: List[str], by_id: Dict[str, Example],
                    n: int) -> List[Example]:
    """Pick the first n ids from the pool and look up their Examples."""
    n = min(n, len(pool))
    return [by_id[eid] for eid in pool[:n]]


def entity_density(examples: List[Example]) -> Dict[str, float]:
    """Entities per token + per-type entity counts. BIO-collapsed counts.

    An 'entity' here is a B-* tag (a span starts).
    """
    n_tokens = 0
    counts: Dict[str, int] = {}
    for ex in examples:
        n_tokens += len(ex.tokens)
        for tag in ex.ner_tags:
            if tag.startswith("B-"):
                t = tag[2:]
                counts[t] = counts.get(t, 0) + 1
    n_ents = sum(counts.values())
    return {
        "n_examples": len(examples),
        "n_tokens": n_tokens,
        "n_entities": n_ents,
        "density": (n_ents / n_tokens) if n_tokens else 0.0,
        "per_type": counts,
    }

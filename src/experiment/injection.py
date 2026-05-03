"""
Seed-controlled target injection pool.

The injection ordering for a given seed is deterministic: same seed → same
order. The ordering is written to disk before any training begins so it is
auditable.
"""

import hashlib
import json
import os
import random
from typing import Dict, List


def build_injection_pool(target_examples: List[Dict], seed: int, out_dir: str) -> List[str]:
    """
    Deterministically shuffle target_examples by id and persist the ordering.

    Writes <out_dir>/injection_order.json with:
        {seed, n, order_hash, ids: [...]}

    Returns the ordered list of ids.
    """
    ids = [ex["id"] for ex in target_examples]
    rng = random.Random(seed)
    rng.shuffle(ids)

    order_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "injection_order.json")
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "n": len(ids), "order_hash": order_hash, "ids": ids},
                  fh, indent=2)
    os.replace(tmp, path)

    return ids


def slice_for_iter(pool_ids: List[str], k: int, step_size: int) -> List[str]:
    """First k * step_size ids from the pool. k=0 → empty list."""
    n = k * step_size
    return pool_ids[:n]


def select_examples(all_examples: List[Dict], ids: List[str]) -> List[Dict]:
    """Return examples in the order specified by `ids`."""
    by_id = {ex["id"]: ex for ex in all_examples}
    return [by_id[i] for i in ids if i in by_id]

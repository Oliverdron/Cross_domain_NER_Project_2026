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
    Deterministically shuffle target examples by id and persist the ordering to disk.

    Seed-determinism guarantee: given the same target_examples and seed, this
    function always produces the same id ordering across runs, machines, and Python
    versions — because it uses a local random.Random instance seeded explicitly
    rather than relying on the global RNG state (which may differ between runs).

    Writes <out_dir>/injection_order.json with:
        {seed, n, order_hash, ids: [...]}

    Args:
        target_examples: list of example dicts, each with an "id" key.
        seed:            integer seed controlling shuffle order.
        out_dir:         directory where injection_order.json is written.
    Returns:
        Ordered list of example ids in the shuffled injection sequence.
    """
    ids = [ex["id"] for ex in target_examples]
    # Local RNG instance isolates pool ordering from the global random state,
    # so calling set_seed() before or after this function does not change the pool.
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


def slice_for_iter(pool_ids: List[str], n_target: int) -> List[str]:
    """
    Return the first n_target ids from the injection pool.

    Args:
        pool_ids: ordered list of ids from build_injection_pool.
        n_target: number of examples to include; 0 returns an empty list.
    Returns:
        Prefix of pool_ids of length min(n_target, len(pool_ids)).
    """
    return pool_ids[:n_target]


def select_examples(all_examples: List[Dict], ids: List[str]) -> List[Dict]:
    """
    Retrieve examples from all_examples in the order given by ids.

    Args:
        all_examples: full list of target example dicts (with "id" keys).
        ids:          ordered list of ids to select (typically from slice_for_iter).
    Returns:
        List of example dicts ordered by ids; ids not found in all_examples are skipped.
    """
    by_id = {ex["id"]: ex for ex in all_examples}
    return [by_id[i] for i in ids if i in by_id]

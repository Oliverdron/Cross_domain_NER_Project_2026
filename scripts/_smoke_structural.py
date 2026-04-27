"""Structural smoke test: exercises data load → injection pool → tokenize →
metric/predictions writers → summary.csv append, without training a model.
Catches integration bugs in everything except the training loop itself.
"""
from __future__ import annotations

import os
import sys
import shutil

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _root)

from src.data import (
    build_injection_pool, entity_density, hash_file, load_split, persist_pool,
    select_injected,
)
from src.evaluate import (
    compute_metrics, write_confusion_matrix, write_predictions_jsonl,
)
from src.logging_utils import append_summary_rows, write_json


def main():
    src_train = load_split("data/universal_train.iob2", "source_train")
    tgt_train = load_split("data/news_train.iob2", "target_train")
    print(f"loaded source={len(src_train)} target={len(tgt_train)}")

    pool = build_injection_pool(tgt_train, seed=42)
    pool2 = build_injection_pool(tgt_train, seed=42)
    assert pool == pool2, "injection pool not deterministic"
    pool_other = build_injection_pool(tgt_train, seed=1337)
    assert pool != pool_other, "different seeds give same order"

    by_id = {ex.id: ex for ex in tgt_train}
    inj = select_injected(pool, by_id, 5)
    assert len(inj) == 5

    # Pretend predictions are gold (perfect model) to verify metric pipeline
    sample = src_train[:20]
    gold = [ex.ner_tags for ex in sample]
    pred = gold
    metrics = compute_metrics(gold, pred)
    assert metrics["entity_f1_micro"] == 1.0 or metrics["entity_f1_micro"] == 0.0  # perfect or no-entities

    out_dir = "/tmp/_smoke_structural"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    write_predictions_jsonl(os.path.join(out_dir, "predictions_test.jsonl"),
                            sample, gold, pred, [[1.0]*len(g) for g in gold])
    write_confusion_matrix(os.path.join(out_dir, "confusion_matrix_test.csv"), gold, pred)

    rows = []
    for i, eval_set in enumerate(["source_test", "target_test", "other_target_test"]):
        row = {
            "exp_name": "smoke_struct", "seed": 42, "iteration": 0,
            "n_target_units": 0, "n_target_examples": 0, "target_fraction": 0.0,
            "eval_set": eval_set,
            "entity_f1_micro": metrics["entity_f1_micro"],
            "entity_f1_macro": metrics["entity_f1_macro"],
            "entity_f1_weighted": metrics["entity_f1_weighted"],
            "entity_precision": metrics["entity_precision"],
            "entity_recall": metrics["entity_recall"],
            "token_f1": metrics["token_f1"],
            "token_acc": metrics["token_acc"],
            "eval_loss": 0.0,
            "train_time_sec": 1.0, "best_epoch": 1, "peak_gpu_mem_mb": 0.0,
        }
        for t, m in metrics["per_type"].items():
            row[f"f1_{t}"] = m["f1"]
            row[f"support_{t}"] = m["support"]
        rows.append(row)
    append_summary_rows(os.path.join(out_dir, "summary.csv"), rows)
    # second append (different iteration) should add rows + preserve old ones
    rows2 = [dict(r, iteration=1, n_target_units=2, n_target_examples=2) for r in rows]
    append_summary_rows(os.path.join(out_dir, "summary.csv"), rows2)
    import csv
    with open(os.path.join(out_dir, "summary.csv")) as f:
        reader = csv.DictReader(f)
        n_rows = sum(1 for _ in reader)
    print(f"summary.csv has {n_rows} rows (expected 6)")
    assert n_rows == 6, f"expected 6 rows, got {n_rows}"

    # Hash test
    h1 = hash_file("data/news_train.iob2")
    h2 = hash_file("data/news_train.iob2")
    assert h1 == h2 and len(h1) == 64
    print("structural smoke test OK")


if __name__ == "__main__":
    main()

"""
Atomic file writers for experiment outputs.

Every writer goes through a tmp-file + os.replace so a crash mid-write never
leaves partial data on disk. summary.csv is append-only (read existing rows,
append, atomic-replace).
"""

import csv
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import __version__ as transformers_version

from config import LABEL_LIST


# --- column schema -----------------------------------------------------------

def _bio_collapsed_types() -> List[str]:
    """Distinct entity types from LABEL_LIST after BIO collapse, sorted."""
    types = set()
    for lab in LABEL_LIST:
        if lab == "O":
            continue
        if "-" in lab:
            types.add(lab.split("-", 1)[1])
        else:
            types.add(lab)
    return sorted(types)


ENTITY_TYPES = _bio_collapsed_types()

SUMMARY_COLUMNS: List[str] = [
    "exp_name", "seed", "iteration",
    "n_blocks", "n_target_examples", "n_source_examples", "train_set_size",
    "target_fraction",
    "eval_set",
    "entity_f1_micro", "entity_f1_macro", "entity_f1_weighted",
    "entity_precision", "entity_recall",
    "token_f1", "token_acc", "eval_loss",
    "delta_f1_vs_iter0",
    *[f"f1_{t}" for t in ENTITY_TYPES],
    *[f"support_{t}" for t in ENTITY_TYPES],
    "best_epoch", "train_time_sec", "peak_gpu_mem_mb",
]


# --- atomic JSON / text writers ---------------------------------------------

def _atomic_write(path: str, data: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
    os.replace(tmp, path)


def write_json(path: str, obj: Any):
    _atomic_write(path, json.dumps(obj, indent=2, default=str))


def write_jsonl(path: str, records: Iterable[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def write_text(path: str, text: str):
    _atomic_write(path, text)


# --- summary.csv -------------------------------------------------------------

def append_summary_row(summary_csv_path: str, row: Dict[str, Any]):
    """
    Append `row` to summary.csv via tmp-rename. Header is written on first call.
    Missing keys are written as empty cells. Unknown keys are ignored.
    """
    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)

    existing_rows: List[Dict[str, str]] = []
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_rows = list(reader)

    safe_row = {col: row.get(col, "") for col in SUMMARY_COLUMNS}

    tmp = summary_csv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({c: r.get(c, "") for c in SUMMARY_COLUMNS})
        writer.writerow(safe_row)
    os.replace(tmp, summary_csv_path)


# --- per-run directories -----------------------------------------------------

def run_root(output_dir: str, exp_name: str) -> Path:
    return Path(output_dir) / exp_name


def init_iter_dir(output_dir: str, exp_name: str, seed: int, iter_idx: int) -> Path:
    p = (run_root(output_dir, exp_name)
         / "seeds" / f"seed_{seed}"
         / f"iter_{iter_idx:03d}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def init_seed_dir(output_dir: str, exp_name: str, seed: int) -> Path:
    p = run_root(output_dir, exp_name) / "seeds" / f"seed_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- config.json snapshot ----------------------------------------------------

def write_config_snapshot(out_path: str, *,
                          cfg_dict: Dict, dataset_hashes: Dict[str, str],
                          git_commit: str, tokenizer) -> None:
    snapshot = {
        "config":          cfg_dict,
        "git_commit":      git_commit,
        "dataset_sha256":  dataset_hashes,
        "tokenizer": {
            "class":       tokenizer.__class__.__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", ""),
            "transformers_version": transformers_version,
        },
        "machine": {
            "platform": platform.platform(),
            "python":   platform.python_version(),
            "torch":    torch.__version__,
            "cuda":     torch.cuda.is_available(),
            "gpu":      torch.cuda.get_device_name() if torch.cuda.is_available() else "",
        },
    }
    write_json(out_path, snapshot)


# --- confusion matrix --------------------------------------------------------

def write_confusion_csv(path: str, labels: List[str], matrix) -> None:
    """matrix: 2-D iterable indexed by labels (rows = gold, cols = pred)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
    os.replace(tmp, path)


# --- summary row builder -----------------------------------------------------

def build_summary_row(*, exp_name: str, seed: int, iteration: int,
                      n_target_examples: int,
                      target_fraction: float, eval_set: str,
                      eval_metrics: Dict, per_type: Dict[str, Dict],
                      train_time_sec: float, best_epoch: int,
                      peak_gpu_mem_mb: int,
                      n_source_examples: int,
                      block_size: int,
                      delta_f1_vs_iter0: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "exp_name":           exp_name,
        "seed":               seed,
        "iteration":          iteration,
        "n_blocks":           n_target_examples // block_size if block_size else 0,
        "n_target_examples":  n_target_examples,
        "n_source_examples":  n_source_examples,
        "train_set_size":     n_source_examples + n_target_examples,
        "target_fraction":    target_fraction,
        "eval_set":           eval_set,
        "entity_f1_micro":    eval_metrics["f1"],
        "entity_f1_macro":    eval_metrics.get("f1_macro", ""),
        "entity_f1_weighted": eval_metrics.get("f1_weighted", ""),
        "entity_precision":   eval_metrics["precision"],
        "entity_recall":      eval_metrics["recall"],
        "token_f1":           eval_metrics.get("token_f1", ""),
        "token_acc":          eval_metrics.get("token_acc", ""),
        "eval_loss":          eval_metrics["loss"],
        "delta_f1_vs_iter0":  delta_f1_vs_iter0,
        "best_epoch":         best_epoch,
        "train_time_sec":     train_time_sec,
        "peak_gpu_mem_mb":    peak_gpu_mem_mb,
    }
    for t in ENTITY_TYPES:
        stats = per_type.get(t, {})
        row[f"f1_{t}"]      = stats.get("f1-score", "")
        row[f"support_{t}"] = stats.get("support", "")
    return row

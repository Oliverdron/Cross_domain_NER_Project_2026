"""summary.csv append, meta.json writer, frozen config snapshot."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from typing import Dict, Iterable, List, Optional

import torch


# Fixed leading columns in summary.csv. Per-type columns (f1_<TYPE>,
# support_<TYPE>) are appended dynamically per row by the caller. The CSV
# is rewritten with a unioned header on every append so that rows added
# later with new types still produce a valid file.
SUMMARY_FIXED_COLS = [
    "exp_name", "seed", "iteration", "n_target_units", "n_target_examples",
    "target_fraction", "eval_set",
    "entity_f1_micro", "entity_f1_macro", "entity_f1_weighted",
    "entity_precision", "entity_recall",
    "token_f1", "token_acc", "eval_loss",
    "train_time_sec", "best_epoch", "peak_gpu_mem_mb",
]


def git_commit_hash(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def machine_info() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_summary_rows(csv_path: str, rows: List[Dict]) -> None:
    """Append rows to summary.csv. Rewrites the whole file with a unioned
    header to handle rows that introduce new per-type columns (different
    eval sets have different entity types). Crash-safe via tmp-file rename.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    existing: List[Dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    all_rows = existing + rows
    extra_cols: List[str] = []
    seen = set(SUMMARY_FIXED_COLS)
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                extra_cols.append(k)
                seen.add(k)
    f1_cols = sorted(c for c in extra_cols if c.startswith("f1_"))
    sup_cols = sorted(c for c in extra_cols if c.startswith("support_"))
    other = [c for c in extra_cols if not c.startswith(("f1_", "support_"))]
    header = SUMMARY_FIXED_COLS + f1_cols + sup_cols + other

    tmp = csv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in header})
    os.replace(tmp, csv_path)


def hash_state_dict(state_dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        t = state_dict[k]
        if hasattr(t, "detach"):
            arr = t.detach().cpu().numpy().tobytes()
        else:
            arr = bytes(t)
        h.update(arr)
    return h.hexdigest()

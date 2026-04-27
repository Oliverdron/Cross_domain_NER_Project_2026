"""Outer loop: for each (seed, iteration), build the training mix, train,
evaluate on every set in eval_sets, and write all logging artifacts.

Usage:
    python scripts/run_experiment.py experiments/config_targetA.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List

import torch
import yaml
from transformers import AutoTokenizer

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.data import (
    Example, build_injection_pool, entity_density, hash_file, load_pool,
    load_split, persist_pool, select_injected,
)
from src.evaluate import (
    compute_metrics, run_inference, write_confusion_matrix,
    write_predictions_jsonl,
)
from src.logging_utils import (
    SUMMARY_FIXED_COLS, append_summary_rows, git_commit_hash,
    hash_state_dict, machine_info, write_json,
)
from src.tokenize_align import tokenize_examples
from src.train import train_one_run


TRUNCATION_WARN_FRAC = 0.10


def load_config(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seeds", [42])
    cfg.setdefault("max_seq_len", 256)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("learning_rate", 5e-5)
    cfg.setdefault("num_epochs", 5)
    cfg.setdefault("early_stopping_patience", 2)
    cfg.setdefault("warmup_ratio", 0.1)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("model_name", "google-bert/bert-base-cased")
    cfg.setdefault("tokenizer_name", cfg["model_name"])
    return cfg


def write_decision_log(run_dir: str, cfg: Dict, target_truncated: int,
                       target_total: int) -> None:
    md = []
    md.append(f"# {cfg['experiment_name']}")
    md.append("")
    md.append(f"- Config: `{cfg.get('_config_path', '')}`")
    md.append(f"- Source: **{cfg['source']['name']}** (unit: {cfg['source'].get('unit','sentence')})")
    md.append(f"- Target: **{cfg['target']['name']}** (unit: {cfg['target'].get('unit','sentence')})")
    md.append(f"- Equivalence ratio (target unit ↔ source unit): see `units_per_step` in config; current run uses 30:1 (1 astro paragraph ≈ 30 EWT/CoNLL sentences) where target is paragraph-split.")
    md.append(f"- max_seq_len: {cfg['max_seq_len']}")
    md.append(f"- Truncated target training examples (iter K, full pool): {target_truncated}/{target_total}"
              f" ({(target_truncated/target_total*100 if target_total else 0):.1f}%)")
    if target_total and target_truncated / target_total > TRUNCATION_WARN_FRAC:
        md.append(f"  - **WARNING**: above the {int(TRUNCATION_WARN_FRAC*100)}% threshold. "
                  f"Sliding windows are out of scope for v1; tail content is dropped.")
    md.append("")
    md.append("## Notes / anomalies")
    md.append("- (fill in as the run progresses)")
    path = os.path.join(run_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    cfg["_config_path"] = os.path.abspath(config_path)
    exp_name = cfg["experiment_name"]
    run_dir = os.path.join(cfg["output_dir"], exp_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Tokenizer (loaded once) ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
    try:
        from transformers import __version__ as transformers_version
    except Exception:
        transformers_version = "unknown"

    # ── Frozen config snapshot ─────────────────────────────────────────────
    dataset_paths = {
        "source_train": cfg["source"]["train"],
        "source_dev":   cfg["source"]["dev"],
        "source_test":  cfg["source"]["test"],
        "target_train": cfg["target"]["train"],
        "target_dev":   cfg["target"]["dev"],
        "target_test":  cfg["target"]["test"],
    }
    for name, path in cfg["eval_sets"].items():
        dataset_paths[f"eval_{name}"] = path
    dataset_hashes = {k: hash_file(v) for k, v in dataset_paths.items()}

    snapshot = {
        "experiment_name": exp_name,
        "config": cfg,
        "git_commit": git_commit_hash(_root),
        "machine": machine_info(),
        "tokenizer_name": cfg["tokenizer_name"],
        "transformers_version": transformers_version,
        "dataset_paths": dataset_paths,
        "dataset_hashes": dataset_hashes,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(os.path.join(run_dir, "config.json"), snapshot)

    # ── Load all data once ─────────────────────────────────────────────────
    print(f"[{exp_name}] Loading datasets …")
    source_train = load_split(cfg["source"]["train"], "source_train")
    target_train = load_split(cfg["target"]["train"], "target_train")
    target_dev   = load_split(cfg["target"]["dev"],   "target_dev")
    print(f"  source_train={len(source_train)}  target_train={len(target_train)}")

    eval_examples: Dict[str, List[Example]] = {
        name: load_split(path, f"eval_{name}")
        for name, path in cfg["eval_sets"].items()
    }

    # ── Tokenize fixed sets once ───────────────────────────────────────────
    src_train_tok = tokenize_examples(source_train, tokenizer, cfg["max_seq_len"])
    target_dev_tok = tokenize_examples(target_dev, tokenizer, cfg["max_seq_len"])
    eval_tok: Dict[str, "TokenizationResult"] = {}
    eval_truncs: Dict[str, Dict] = {}
    for name, exs in eval_examples.items():
        tr = tokenize_examples(exs, tokenizer, cfg["max_seq_len"])
        eval_tok[name] = tr
        eval_truncs[name] = {"n_truncated": tr.n_truncated, "n_total": tr.n_total}
        if tr.n_total and tr.n_truncated / tr.n_total > TRUNCATION_WARN_FRAC:
            warnings.warn(
                f"[{exp_name}] eval set '{name}': "
                f"{tr.n_truncated}/{tr.n_total} ({tr.n_truncated/tr.n_total*100:.1f}%) "
                f"examples truncated at max_seq_len={cfg['max_seq_len']}."
            )

    # ── Iteration plan ─────────────────────────────────────────────────────
    units_per_step = int(cfg["target"]["units_per_step"])
    n_iterations = int(cfg["target"]["n_iterations"])
    target_unit = cfg["target"].get("unit", "sentence")
    # 1 unit == 1 example (whether sentence or paragraph) — the budget
    # equivalence ratio (30:1 for paragraphs) is what the user uses to
    # *compare* across configs. Within a single config we just step by
    # n examples per iteration.
    target_by_id = {ex.id: ex for ex in target_train}

    # Smoke-test: we save predictions on iter 0 with a fixed seed → write
    # a hash of (gold + pred tag sequences) in metrics.json as a
    # reproducibility checksum.
    write_decision_log(run_dir, cfg, 0, len(target_train))

    target_truncated_so_far = 0

    for seed in cfg["seeds"]:
        seed_dir = os.path.join(run_dir, "seeds", f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        pool = build_injection_pool(target_train, seed)
        persist_pool(pool, os.path.join(seed_dir, "injection_pool.txt"))

        for k in range(n_iterations):
            iter_dir = os.path.join(seed_dir, f"iter_{k:03d}")
            os.makedirs(iter_dir, exist_ok=True)
            ckpt_dir = os.path.join(iter_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)

            n_target_units = min(k * units_per_step, len(pool))
            injected = select_injected(pool, target_by_id, n_target_units)

            with open(os.path.join(iter_dir, "added_target_ids.txt"), "w", encoding="utf-8") as f:
                for ex in injected:
                    f.write(ex.id + "\n")

            mix = list(source_train) + list(injected)
            mix_tok = tokenize_examples(mix, tokenizer, cfg["max_seq_len"])
            target_truncated_in_mix = max(0, mix_tok.n_truncated - src_train_tok.n_truncated)
            target_truncated_so_far = max(target_truncated_so_far, target_truncated_in_mix)

            # Densities for meta.json
            mix_density = entity_density(mix)
            eval_density = {name: entity_density(exs) for name, exs in eval_examples.items()}
            target_dev_density = entity_density(target_dev)

            print(f"\n[{exp_name}] seed={seed} iter={k}  source={len(source_train)}  "
                  f"target_inj={len(injected)}  total={len(mix)}")

            # ── Train ──────────────────────────────────────────────────────
            tr = train_one_run(
                train_dataset=mix_tok.dataset,
                dev_dataset=target_dev_tok.dataset,
                tokenizer=tokenizer,
                model_name=cfg["model_name"],
                device=device,
                seed=seed,
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                num_epochs=cfg["num_epochs"],
                early_stopping_patience=cfg["early_stopping_patience"],
                warmup_ratio=cfg["warmup_ratio"],
                weight_decay=cfg["weight_decay"],
            )

            # Save checkpoint (weights only)
            ckpt_path = os.path.join(ckpt_dir, "state_dict.pt")
            torch.save(tr.best_state_dict, ckpt_path)
            ckpt_hash = hash_state_dict(tr.best_state_dict)

            # train_log.jsonl
            with open(os.path.join(iter_dir, "train_log.jsonl"), "w", encoding="utf-8") as f:
                for row in tr.train_log:
                    f.write(json.dumps(row) + "\n")

            # ── Eval on every eval set ─────────────────────────────────────
            from transformers import AutoModelForTokenClassification
            from src.train import build_model
            eval_model = build_model(cfg["model_name"]).to(device)
            eval_model.load_state_dict(tr.best_state_dict)

            all_metrics: Dict[str, Dict] = {}
            all_per_type: Dict[str, Dict] = {}
            summary_rows: List[Dict] = []
            for name, tr_tok in eval_tok.items():
                inf = run_inference(eval_model, tr_tok.dataset, tokenizer,
                                    device, cfg["batch_size"])
                metrics = compute_metrics(inf["gold_tags"], inf["pred_tags"])
                metrics["eval_loss"] = inf["eval_loss"]

                # Predictions JSONL + confusion matrix (token-level, BIO-collapsed)
                write_predictions_jsonl(
                    os.path.join(iter_dir, f"predictions_{name}.jsonl"),
                    eval_examples[name], inf["gold_tags"], inf["pred_tags"], inf["pred_probs"],
                )
                write_confusion_matrix(
                    os.path.join(iter_dir, f"confusion_matrix_{name}.csv"),
                    inf["gold_tags"], inf["pred_tags"],
                )

                row = {
                    "exp_name": exp_name,
                    "seed": seed,
                    "iteration": k,
                    "n_target_units": n_target_units,
                    "n_target_examples": len(injected),
                    "target_fraction": len(injected) / max(len(mix), 1),
                    "eval_set": name,
                    "entity_f1_micro": metrics["entity_f1_micro"],
                    "entity_f1_macro": metrics["entity_f1_macro"],
                    "entity_f1_weighted": metrics["entity_f1_weighted"],
                    "entity_precision": metrics["entity_precision"],
                    "entity_recall": metrics["entity_recall"],
                    "token_f1": metrics["token_f1"],
                    "token_acc": metrics["token_acc"],
                    "eval_loss": metrics["eval_loss"],
                    "train_time_sec": tr.train_time_sec,
                    "best_epoch": tr.best_epoch,
                    "peak_gpu_mem_mb": tr.peak_gpu_mem_mb,
                }
                for t, m in metrics["per_type"].items():
                    row[f"f1_{t}"] = m["f1"]
                    row[f"support_{t}"] = m["support"]
                summary_rows.append(row)

                all_metrics[name] = {kk: vv for kk, vv in metrics.items() if kk != "per_type"}
                all_per_type[name] = metrics["per_type"]

            # metrics.json + per_type_metrics.json
            write_json(os.path.join(iter_dir, "metrics.json"), all_metrics)
            write_json(os.path.join(iter_dir, "per_type_metrics.json"), all_per_type)

            # meta.json
            meta = {
                "exp_name": exp_name,
                "seed": seed,
                "iteration": k,
                "n_source": len(source_train),
                "n_target_injected": len(injected),
                "n_target_units": n_target_units,
                "target_unit": target_unit,
                "target_fraction": len(injected) / max(len(mix), 1),
                "target_truncated_in_mix": target_truncated_in_mix,
                "n_examples_total": len(mix),
                "best_epoch": tr.best_epoch,
                "best_dev_f1": tr.best_dev_f1,
                "train_time_sec": tr.train_time_sec,
                "peak_gpu_mem_mb": tr.peak_gpu_mem_mb,
                "checkpoint_sha256": ckpt_hash,
                "train_mix_density": mix_density,
                "target_dev_density": target_dev_density,
                "eval_density": eval_density,
                "eval_truncations": eval_truncs,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            write_json(os.path.join(iter_dir, "meta.json"), meta)

            # Append summary.csv after each (seed, iter) finishes
            append_summary_rows(os.path.join(run_dir, "summary.csv"), summary_rows)

            del eval_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Update README with the actual truncation stats now that we have them
    write_decision_log(run_dir, cfg, target_truncated_so_far, len(target_train))

    print(f"\n[{exp_name}] Done. Results under {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to experiments/<name>/config.yaml")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

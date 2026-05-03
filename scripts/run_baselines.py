"""
Three single-domain baselines (train-once, not iterative):
  - ewt_only   : train on EWT,   eval on ewt_dev / conll_test / astro_test
  - conll_only : train on CoNLL, eval on ewt_dev / conll_test / astro_test
  - astro_only : train on Astro (paragraphs), eval on ewt_dev / conll_test / astro_test

Each writes to runs/baselines/<name>/summary.csv using the SAME schema as the
iterative experiment so plots can overlay.

Usage:
    python scripts/run_baselines.py [--baselines ewt_only conll_only astro_only]
                                    [--seeds 42 123 456]

The script does not run automatically on import.
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import (
    parse_iob2,
    prepare_split,
    make_dataloader,
    entity_density,
)
from model import build_model
from trainer import set_seed as seed_everything, train as trainer_train

from src.experiment.evaluation import full_evaluate
from src.experiment.logging_io import (
    append_summary_row,
    build_summary_row,
    init_iter_dir,
    run_root,
    write_confusion_csv,
    write_json,
    write_jsonl,
)


# --- baseline specs ---------------------------------------------------------

BASELINES: Dict[str, Dict] = {
    "ewt_only": {
        "train_path": "data/universal_train.iob2",
        "dev_path":   "data/universal_dev.iob2",
        "ds_name":    "ewt",
        "unit":       "sentence",
        "max_seq_len": 256,
        "batch_size":  16,
        "token_col":  1, "tag_col":  2,
    },
    "conll_only": {
        "train_path": "data/news_train.iob2",
        "dev_path":   "data/news_dev.iob2",
        "ds_name":    "conll",
        "unit":       "sentence",
        "max_seq_len": 256,
        "batch_size":  16,
        "token_col":  0, "tag_col":  1,
    },
    "astro_only": {
        "train_path": "data/astro_train.iob2",
        "dev_path":   "data/astro_dev.iob2",
        "ds_name":    "astro",
        "unit":       "paragraph",
        "max_seq_len": 512,
        "batch_size":  4,
        "token_col":  0, "tag_col":  1,
    },
}

# eval sets shared with the iterative runs (EWT test masked → use EWT dev)
EVAL_SETS: List[Dict] = [
    {"name": "ewt_dev",    "path": "data/universal_dev.iob2", "token_col": 1, "tag_col": 2, "unit": "sentence"},
    {"name": "conll_test", "path": "data/news_test.iob2",     "token_col": 0, "tag_col": 1, "unit": "sentence"},
    {"name": "astro_test", "path": "data/astro_test.iob2",    "token_col": 0, "tag_col": 1, "unit": "paragraph"},
]


# --- shared training defaults ----------------------------------------------

DEFAULTS = SimpleNamespace(
    model_name="google-bert/bert-base-cased",
    tokenizer_name="google-bert/bert-base-cased",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    num_epochs=100,
    early_stopping_patience=3,
)


def _load(path, token_col, tag_col, unit):
    return parse_iob2(path, token_col=token_col, tag_col=tag_col, unit=unit)


def _build_eval_loader(examples, tokenizer, max_seq_len, batch_size):
    ds, n_trunc = prepare_split(examples, tokenizer, max_seq_len,
                                return_truncation_count=True)
    return make_dataloader(ds, tokenizer, batch_size, shuffle=False), n_trunc


def run_one_baseline(name: str, spec: Dict, seeds: List[int],
                     output_dir: str, device: torch.device,
                     debug: bool = False):
    print(f"\n══════════ baseline: {name} ══════════")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULTS.tokenizer_name, use_fast=True)

    train_examples = _load(spec["train_path"], spec["token_col"], spec["tag_col"], spec["unit"])
    dev_examples   = _load(spec["dev_path"],   spec["token_col"], spec["tag_col"], spec["unit"])

    if debug:
        train_examples = train_examples[:max(1, int(len(train_examples) * 0.001))]
        dev_examples   = dev_examples[:max(1, int(len(dev_examples) * 0.001))]

    print(f"  train: {len(train_examples)}, dev: {len(dev_examples)}")
    print(f"  entity_density train: {entity_density(train_examples):.4f}")

    eval_examples_map: Dict[str, List[Dict]] = {}
    eval_loaders:      Dict[str, "torch.utils.data.DataLoader"] = {}
    eval_ids_map:      Dict[str, List[str]] = {}
    eval_tokens_map:   Dict[str, List[List[str]]] = {}
    for e in EVAL_SETS:
        ex = _load(e["path"], e["token_col"], e["tag_col"], e["unit"])
        if debug:
            ex = ex[:max(1, int(len(ex) * 0.001))]
        loader, n_t = _build_eval_loader(ex, tokenizer,
                                         spec["max_seq_len"], spec["batch_size"])
        eval_examples_map[e["name"]] = ex
        eval_loaders[e["name"]]      = loader
        eval_ids_map[e["name"]]      = [x["id"]     for x in ex]
        eval_tokens_map[e["name"]]   = [x["tokens"] for x in ex]
        n = len(ex)
        pct = (n_t / n * 100) if n else 0.0
        print(f"  eval {e['name']:<11} truncated {n_t}/{n} ({pct:.1f}%)")
        if e["unit"] == "paragraph" and pct > 10.0:
            print(f"  !! WARNING: >10% astro paragraphs truncated for {e['name']}")

    # tokenise dev once for early stopping
    dev_loader, n_dev_trunc = _build_eval_loader(dev_examples, tokenizer,
                                                 spec["max_seq_len"], spec["batch_size"])

    train_ds, n_train_trunc = prepare_split(train_examples, tokenizer,
                                            spec["max_seq_len"],
                                            return_truncation_count=True)

    # baseline-level dir + shared summary.csv
    base_root = Path(output_dir) / "baselines" / name
    base_root.mkdir(parents=True, exist_ok=True)
    summary_csv = base_root / "summary.csv"

    write_json(str(base_root / "config.json"), {
        "baseline":     name,
        "spec":         spec,
        "defaults":     vars(DEFAULTS),
        "eval_sets":    [e["name"] for e in EVAL_SETS],
        "n_train":      len(train_examples),
        "n_dev":        len(dev_examples),
        "trunc_train":  n_train_trunc,
        "trunc_dev":    n_dev_trunc,
    })

    for seed in seeds:
        print(f"\n── {name}, seed {seed} ──")
        seed_everything(seed)
        iter_dir = init_iter_dir(output_dir, f"baselines/{name}", seed, 0)

        train_loader = make_dataloader(train_ds, tokenizer, spec["batch_size"], shuffle=True)

        # FRESH MODEL — pretrained checkpoint each seed.
        model = build_model(DEFAULTS.model_name).to(device)

        t_args = SimpleNamespace(
            lr=DEFAULTS.learning_rate,
            weight_decay=DEFAULTS.weight_decay,
            warmup_ratio=DEFAULTS.warmup_ratio,
            epochs=DEFAULTS.num_epochs,
            output_dir=str(iter_dir),
        )
        train_result = trainer_train(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            args=t_args,
            early_stopping_patience=DEFAULTS.early_stopping_patience,
            train_log_path=str(iter_dir / "train_log.jsonl"),
            best_model_dir=str(iter_dir / "_best_hf"),
            dev_desc=f"{name} dev (seed={seed})",
        )

        model = AutoModelForTokenClassification.from_pretrained(
            train_result["best_model_dir"],
        ).to(device)
        shutil.rmtree(train_result["best_model_dir"])

        metrics_per_set: Dict[str, Dict] = {}
        for ev_name, loader in eval_loaders.items():
            metrics_per_set[ev_name] = full_evaluate(
                model, loader, device, ev_name,
                example_ids=eval_ids_map[ev_name],
                example_tokens=eval_tokens_map[ev_name],
            )

        write_json(str(iter_dir / "meta.json"), {
            "iteration":          0,
            "seed":               seed,
            "baseline":           name,
            "n_target_examples":  0,
            "n_target_units":     0,
            "target_fraction":    "",       # NaN-equivalent in CSV
            "train_time_sec":     train_result["train_time_sec"],
            "best_epoch":         train_result["best_epoch"],
            "best_dev_f1":        train_result["best_dev_f1"],
            "epochs_run":         train_result["epochs_run"],
            "peak_gpu_mem_mb":    train_result["peak_gpu_mem_mb"],
            "truncation_count":   n_train_trunc,
        })

        write_json(str(iter_dir / "metrics.json"), {
            ev: {k: m[k] for k in
                 ("f1", "f1_macro", "f1_weighted", "precision", "recall",
                  "loss", "token_acc", "token_f1")}
            for ev, m in metrics_per_set.items()
        })
        write_json(str(iter_dir / "per_type_metrics.json"),
                   {ev: m["per_type"] for ev, m in metrics_per_set.items()})

        for ev, m in metrics_per_set.items():
            write_confusion_csv(str(iter_dir / f"confusion_matrix_{ev}.csv"),
                                m["confusion_labels"], m["confusion_matrix"])
            write_jsonl(str(iter_dir / f"predictions_{ev}.jsonl"), m["predictions"])

        for ev, m in metrics_per_set.items():
            row = build_summary_row(
                exp_name=f"baseline_{name}",
                seed=seed,
                iteration=0,
                n_target_examples=0,
                target_fraction="",        # NaN-equivalent
                eval_set=ev,
                eval_metrics=m,
                per_type=m["per_type"],
                train_time_sec=train_result["train_time_sec"],
                best_epoch=train_result["best_epoch"],
                peak_gpu_mem_mb=train_result["peak_gpu_mem_mb"],
                n_source_examples=len(train_examples),
                block_size=1,
                delta_f1_vs_iter0=0.0,
            )
            append_summary_row(str(summary_csv), row)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+",
                        default=list(BASELINES.keys()),
                        choices=list(BASELINES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        print("⚠️  DEBUG MODE: using 0.1% of data, 1 seed, 5 iterations max. Do not use for real experiments.")
        args.seeds = [42]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")

    for name in args.baselines:
        run_one_baseline(name, BASELINES[name], args.seeds, args.output_dir, device,
                         debug=args.debug)


if __name__ == "__main__":
    main()

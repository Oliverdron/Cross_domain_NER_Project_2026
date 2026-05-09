"""
Iterative cross-domain NER experiment runner.

Usage:
    python scripts/run_experiment.py --config experiments/config_conll.yaml

The script does not run automatically on import. Training only starts when
this file is executed as __main__.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

# repo root on sys.path so plain `import data` etc. works regardless of cwd
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

from src.experiment.config_loader import (
    ExperimentConfig,
    collect_dataset_hashes,
    git_commit_hash,
    load_config,
)
from src.experiment.injection import (
    build_injection_pool,
    select_examples,
    slice_for_iter,
)
from src.experiment.evaluation import full_evaluate
from src.experiment.logging_io import (
    append_summary_row,
    build_summary_row,
    init_iter_dir,
    init_seed_dir,
    run_root,
    write_confusion_csv,
    write_config_snapshot,
    write_json,
    write_jsonl,
    write_text,
)


# ----- IOB2 column rules per dataset name -----------------------------------
_IOB2_COLS = {
    "ewt":   dict(token_col=1, tag_col=2),
    "conll": dict(token_col=0, tag_col=1),
    "astro": dict(token_col=0, tag_col=1),
    "wiesp": dict(token_col=0, tag_col=1),
}


def _cols_for(name: str) -> Dict[str, int]:
    return _IOB2_COLS.get(name.lower(), dict(token_col=0, tag_col=1))


def _load_examples(path: str, dataset_name: str, unit: str):
    return parse_iob2(path, unit=unit, **_cols_for(dataset_name))


def _build_eval_loader(examples, tokenizer, max_seq_len, batch_size, device):
    ds, n_trunc = prepare_split(examples, tokenizer, max_seq_len,
                                return_truncation_count=True)
    loader = make_dataloader(ds, tokenizer, batch_size, shuffle=False)
    return loader, n_trunc


def _print_truncation(name: str, examples, n_trunc: int, unit: str):
    n = len(examples)
    pct = (n_trunc / n * 100) if n else 0.0
    print(f"  {name:<22} truncated: {n_trunc}/{n} ({pct:.1f}%)")
    if unit == "paragraph" and pct > 10.0:
        print(f"  !! WARNING: >10% astro paragraphs truncated for {name} "
              f"({n_trunc}/{n}). Consider raising max_seq_len.")


def _trainer_args(cfg: ExperimentConfig, output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        epochs=cfg.num_epochs,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seeds = list(cfg.seeds)
    schedule = list(cfg.schedule)
    if args.debug:
        print("⚠️  DEBUG MODE: using 0.1% of data, 1 seed, 5 iterations max. Do not use for real experiments.")
        seeds = [42]
        schedule = schedule[:5]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")

    # ── Tokenizer (single instance, reused across iterations) ────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

    # ── Load source + target + eval splits ───────────────────────────────────
    print("\n── Loading datasets ─────────────────────────────────────")
    source_train = _load_examples(cfg.source.train, cfg.source.name, cfg.source.unit)
    target_train = _load_examples(cfg.target.train, cfg.target.name, cfg.target.unit)
    target_dev   = _load_examples(cfg.target.dev,   cfg.target.name, cfg.target.unit)

    eval_examples = {
        e.name: _load_examples(e.path, _eval_dataset_name(e.name), _eval_unit(cfg, e.name))
        for e in cfg.eval_sets
    }

    if args.debug:
        source_train  = source_train[:max(1, int(len(source_train) * 0.001))]
        target_dev    = target_dev[:max(1, int(len(target_dev) * 0.001))]
        eval_examples = {k: v[:max(1, int(len(v) * 0.001))] for k, v in eval_examples.items()}

    print(f"  source.train ({cfg.source.name}, {cfg.source.unit}): {len(source_train)}")
    print(f"  target.train ({cfg.target.name}, {cfg.target.unit}): {len(target_train)}")
    print(f"  target.dev   ({cfg.target.name}, {cfg.target.unit}): {len(target_dev)}")
    for name, ex in eval_examples.items():
        print(f"  eval.{name:<10} {len(ex)}")

    # ── Entity density per loaded split ──────────────────────────────────────
    print("\n── Entity density ───────────────────────────────────────")
    print(f"  source.train: {entity_density(source_train):.4f}")
    print(f"  target.train: {entity_density(target_train):.4f}")
    print(f"  target.dev  : {entity_density(target_dev):.4f}")
    for name, ex in eval_examples.items():
        print(f"  eval.{name:<10} {entity_density(ex):.4f}")

    # ── Tokenise eval splits + target dev ONCE (don't change across iters) ──
    print("\n── Tokenising eval + dev splits ─────────────────────────")
    target_dev_loader, n_dev_trunc = _build_eval_loader(
        target_dev, tokenizer, cfg.max_seq_len, cfg.batch_size, device,
    )
    _print_truncation("target.dev", target_dev, n_dev_trunc, cfg.target.unit)

    eval_loaders: Dict[str, "torch.utils.data.DataLoader"] = {}
    eval_ids:     Dict[str, List[str]] = {}
    eval_tokens:  Dict[str, List[List[str]]] = {}
    for name, ex in eval_examples.items():
        loader, n_t = _build_eval_loader(ex, tokenizer, cfg.max_seq_len,
                                         cfg.batch_size, device)
        eval_loaders[name] = loader
        eval_ids[name]     = [e["id"]     for e in ex]
        eval_tokens[name]  = [e["tokens"] for e in ex]
        _print_truncation(f"eval.{name}", ex, n_t, _eval_unit(cfg, name))

    # ── Run-level config snapshot ────────────────────────────────────────────
    exp_dir = run_root(cfg.output_dir, cfg.experiment_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = exp_dir / "summary.csv"

    write_config_snapshot(
        str(exp_dir / "config.json"),
        cfg_dict=cfg.to_dict(),
        dataset_hashes=collect_dataset_hashes(cfg),
        git_commit=git_commit_hash(),
        tokenizer=tokenizer,
    )

    # ── Iterate over seeds × iterations ──────────────────────────────────────
    n_source = len(source_train)
    iter0_f1: Dict[tuple, float] = {}

    for seed in seeds:
        print(f"\n══════════ seed {seed} ══════════")
        seed_everything(seed)

        seed_dir = init_seed_dir(cfg.output_dir, cfg.experiment_name, seed)
        pool_ids = build_injection_pool(target_train, seed, str(seed_dir))

        if args.debug:
            pool_ids = pool_ids[:max(1, int(len(pool_ids) * 0.001))]

        for k, n_target in enumerate(schedule):
            print(f"\n── seed {seed}, iter {k} ─────────────────────────────")
            iter_dir = init_iter_dir(cfg.output_dir, cfg.experiment_name, seed, k)

            # ── Resume: skip if this (seed, iter) already completed ────────
            # metrics.json is written near the end of the iter body; if it
            # exists, the iteration produced its outputs (per-iter files +
            # summary.csv rows). Safe to skip on resubmission after a TIMEOUT.
            if (iter_dir / "metrics.json").exists():
                print(f"  ⏩ already complete (metrics.json present) — skipping")
                continue

            target_chunk_ids = slice_for_iter(pool_ids, n_target)
            target_chunk     = select_examples(target_train, target_chunk_ids)
            mix              = source_train + target_chunk

            print(f"  mix size: {n_source} source + {len(target_chunk)} target "
                  f"= {len(mix)}")

            train_ds, n_mix_trunc = prepare_split(
                mix, tokenizer, cfg.max_seq_len,
                return_truncation_count=True,
            )
            train_loader = make_dataloader(
                train_ds, tokenizer, cfg.batch_size, shuffle=True,
            )

            # FRESH MODEL — DO NOT MOVE OUTSIDE LOOP. Re-initialised from the
            # pretrained checkpoint at every iteration.
            model = build_model(cfg.model_name).to(device)

            t_args = _trainer_args(cfg, str(iter_dir))
            t_args.output_dir = str(iter_dir)  # best checkpoint path
            train_result = trainer_train(
                model=model,
                train_loader=train_loader,
                dev_loader=target_dev_loader,
                device=device,
                args=t_args,
                early_stopping_patience=cfg.early_stopping_patience,
                train_log_path=str(iter_dir / "train_log.jsonl"),
                best_model_dir=str(iter_dir / "_best_hf"),
                dev_desc=f"target.dev (seed={seed}, iter={k})",
            )

            # ── Reload best HF checkpoint, evaluate everywhere ───────────────
            model = AutoModelForTokenClassification.from_pretrained(
                train_result["best_model_dir"],
            ).to(device)
            shutil.rmtree(train_result["best_model_dir"])

            metrics_per_set: Dict[str, Dict] = {}
            for name, loader in eval_loaders.items():
                metrics_per_set[name] = full_evaluate(
                    model, loader, device, name,
                    example_ids=eval_ids[name],
                    example_tokens=eval_tokens[name],
                )

            # ── Persist artifacts ────────────────────────────────────────────
            target_fraction = (n_target / (n_source + n_target)) if (n_source + n_target) else 0.0

            write_json(str(iter_dir / "meta.json"), {
                "iteration":               k,
                "seed":                    seed,
                "n_target_examples":       n_target,
                "n_target_units":          n_target,
                "target_fraction":         target_fraction,
                "train_time_sec":          train_result["train_time_sec"],
                "best_epoch":              train_result["best_epoch"],
                "best_dev_f1":             train_result["best_dev_f1"],
                "epochs_run":              train_result["epochs_run"],
                "peak_gpu_mem_mb":         train_result["peak_gpu_mem_mb"],
                "truncation_count":        n_mix_trunc,
                "entity_density_train_mix": entity_density(mix),
                "target_unit":             cfg.target.unit,
            })

            write_json(str(iter_dir / "metrics.json"), {
                name: {kk: m[kk] for kk in
                       ("f1", "f1_macro", "f1_weighted", "precision", "recall",
                        "loss", "token_acc", "token_f1")}
                for name, m in metrics_per_set.items()
            })

            write_json(str(iter_dir / "per_type_metrics.json"), {
                name: m["per_type"] for name, m in metrics_per_set.items()
            })

            for name, m in metrics_per_set.items():
                write_confusion_csv(
                    str(iter_dir / f"confusion_matrix_{name}.csv"),
                    m["confusion_labels"], m["confusion_matrix"],
                )
                write_jsonl(
                    str(iter_dir / f"predictions_{name}.jsonl"),
                    m["predictions"],
                )

            write_text(str(iter_dir / "added_target_ids.txt"),
                       "\n".join(target_chunk_ids) + ("\n" if target_chunk_ids else ""))

            # ── summary.csv: one row per eval set ────────────────────────────
            for name, m in metrics_per_set.items():
                if k == 0:
                    iter0_f1[(seed, name)] = m["f1"]
                delta = m["f1"] - iter0_f1.get((seed, name), m["f1"])
                row = build_summary_row(
                    exp_name=cfg.experiment_name,
                    seed=seed,
                    iteration=k,
                    n_target_examples=n_target,
                    target_fraction=target_fraction,
                    eval_set=name,
                    eval_metrics=m,
                    per_type=m["per_type"],
                    train_time_sec=train_result["train_time_sec"],
                    best_epoch=train_result["best_epoch"],
                    peak_gpu_mem_mb=train_result["peak_gpu_mem_mb"],
                    n_source_examples=n_source,
                    block_size=cfg.block_size,
                    delta_f1_vs_iter0=delta,
                )
                append_summary_row(str(summary_csv), row)



def _eval_dataset_name(eval_set_name: str) -> str:
    """Map eval-set key (e.g. 'astro_test') back to a dataset name for column rules."""
    if eval_set_name.startswith("ewt"):
        return "ewt"
    if eval_set_name.startswith("conll"):
        return "conll"
    if eval_set_name.startswith("astro") or eval_set_name.startswith("wiesp"):
        return "astro"
    return "conll"  # 2-col default


def _eval_unit(cfg: ExperimentConfig, eval_set_name: str) -> str:
    """Astro evals use the same unit as cfg.target.unit if target is astro,
    otherwise paragraph (the file is always paragraph-blocked anyway)."""
    if eval_set_name.startswith("astro") or eval_set_name.startswith("wiesp"):
        return "paragraph"
    return "sentence"


if __name__ == "__main__":
    main()

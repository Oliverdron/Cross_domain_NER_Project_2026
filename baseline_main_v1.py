"""
baseline_main_v1.py — Single-run cross-domain NER baseline entry point.

Trains on EWT, evaluates on EWT / CoNLL / WIESP dev sets, and writes EWT test
predictions to file for LearnIT submission. Superseded by scripts/run_baselines.py
for multi-seed, multi-dataset experiments.
"""

import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import get_args
from data import load_all_datasets, prepare_split, make_dataloader, save_predictions
from model import build_model
from trainer import set_seed as seed_everything, train, evaluate


def print_results_table(results: dict) -> None:
    """
    Print a formatted table of F1, Precision, and Recall for each evaluated split.

    Args:
        results: dict mapping split name to metrics dict (keys: f1, precision, recall).
    """
    print("\n" + "=" * 60)
    print(f"{'Split':<30} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("=" * 60)

    # Compute precision and recall from the seqeval report string
    # (seqeval f1_score only returns F1 — parse report for P/R)
    
    for split_name, metrics in results.items():
        print(f"{split_name:<30} {metrics['f1']:>7.4f} "
              f"{metrics.get('precision', 0.0):>10.4f} "
              f"{metrics.get('recall', 0.0):>8.4f}")
    print("=" * 60)


def main() -> None:
    """
    End-to-end single-seed baseline: load data, train on EWT, evaluate cross-domain,
    and save EWT test predictions.
    """
    args = get_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    seed_everything(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading datasets ─────────────────────────────────────")
    datasets = load_all_datasets(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print("\n── Tokenising ───────────────────────────────────────────")
    ewt_train_ds   = prepare_split(datasets["ewt"]["train"],   tokenizer, args.max_length)
    ewt_dev_ds     = prepare_split(datasets["ewt"]["dev"],     tokenizer, args.max_length)
    ewt_test_ds    = prepare_split(datasets["ewt"]["test"],    tokenizer, args.max_length)
    conll_dev_ds   = prepare_split(datasets["conll"]["dev"],   tokenizer, args.max_length)
    conll_test_ds  = prepare_split(datasets["conll"]["test"],  tokenizer, args.max_length)
    wiesp_dev_ds   = prepare_split(datasets["wiesp"]["dev"],   tokenizer, args.max_length)
    wiesp_test_ds  = prepare_split(datasets["wiesp"]["test"],  tokenizer, args.max_length)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader      = make_dataloader(ewt_train_ds,  tokenizer, args.batch_size, shuffle=True)
    ewt_dev_loader    = make_dataloader(ewt_dev_ds,    tokenizer, args.batch_size)
    ewt_test_loader   = make_dataloader(ewt_test_ds,   tokenizer, args.batch_size)
    conll_dev_loader  = make_dataloader(conll_dev_ds,  tokenizer, args.batch_size)
    conll_test_loader = make_dataloader(conll_test_ds, tokenizer, args.batch_size)
    wiesp_dev_loader  = make_dataloader(wiesp_dev_ds,  tokenizer, args.batch_size)
    wiesp_test_loader = make_dataloader(wiesp_test_ds, tokenizer, args.batch_size)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n── Training on EWT ──────────────────────────────────────")
    model = build_model(args.model_name)
    model.to(device)

    train_result = train(
        model=model,
        train_loader=train_loader,
        dev_loader=ewt_dev_loader,
        device=device,
        args=args,
    )
    best_model_dir = train_result["best_model_dir"]

    # ── Load best checkpoint ──────────────────────────────────────────────────
    print(f"\n── Loading best model from {best_model_dir} ─────────────")
    model = AutoModelForTokenClassification.from_pretrained(best_model_dir)
    model.to(device)

    # ── Cross-domain evaluation ───────────────────────────────────────────────
    print("\n── Evaluating on all splits ─────────────────────────────")

    eval_splits = [
        ("EWT dev (in-domain)",        ewt_dev_loader),
        ("CoNLL-2003 dev (similar)",   conll_dev_loader),
        ("WIESP-2022 dev (different)", wiesp_dev_loader),
    ]

    if args.final_eval:
        eval_splits += [
            ("CoNLL-2003 test (similar)",   conll_test_loader),
            ("WIESP-2022 test (different)", wiesp_test_loader),
        ]


    results = {}
    for split_name, loader in eval_splits:
        metrics = evaluate(model, loader, device, desc=split_name)
        print(f"\n{split_name}")
        print(metrics["report"])
        results[split_name] = metrics

    # ── EWT test predictions → file ───────────────────────────────────────────
    # The EWT test tags are masked — we save predictions to upload to LearnIT.
    print("\n── Saving EWT test predictions ──────────────────────────")
    ewt_test_metrics = evaluate(model, ewt_test_loader, device, desc="EWT test")
    pred_path = os.path.join(args.output_dir, "ewt_test_predictions.iob2")
    save_predictions(
        sentences=datasets["ewt"]["test"],
        predictions=ewt_test_metrics["predictions"],
        output_path=pred_path,
    )

    # ── Final results table ───────────────────────────────────────────────────
    print_results_table(results)
    print(f"\nEWT test predictions saved to: {pred_path}")
    print("Upload that file to LearnIT to get your official EWT test F1.\n")


if __name__ == "__main__":
    main()

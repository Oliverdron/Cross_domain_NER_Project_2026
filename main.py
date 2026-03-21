"""
main.py
-------
Entry point for the cross-domain NER baseline.

Usage:
    python main.py [--data_dir data_iob2] [--output_dir outputs] [--epochs 5] ...

Full argument list: see config.py or run  python main.py --help

What it does:
    1. Loads EWT, CoNLL-2003, and WIESP-2022 from local .iob2 files
    2. Fine-tunes BERT on EWT train, picking the best checkpoint by EWT dev F1
    3. Evaluates the best model on all dev and test splits
    4. Saves EWT test predictions to outputs/ewt_test_predictions.iob2
       (upload this file to LearnIT for official scoring)
    5. Prints a results table across all six evaluation splits
"""

import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed

from config import get_args
from data import load_all_datasets, prepare_split, make_dataloader, save_predictions
from model import build_model
from trainer import set_seed as seed_everything, train, evaluate


def print_results_table(results: dict):
    print("\n" + "=" * 60)
    print(f"{'Split':<30} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("=" * 60)

    # Compute precision and recall from the seqeval report string
    # (seqeval f1_score only returns F1 — parse report for P/R)
    from seqeval.metrics import precision_score, recall_score
    for split_name, metrics in results.items():
        print(f"{split_name:<30} {metrics['f1']:>7.4f} "
              f"{metrics.get('precision', 0.0):>10.4f} "
              f"{metrics.get('recall', 0.0):>8.4f}")
    print("=" * 60)


def main():
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

    best_model_dir = train(
        model=model,
        train_loader=train_loader,
        dev_loader=ewt_dev_loader,
        device=device,
        args=args,
    )

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

    from seqeval.metrics import precision_score, recall_score, f1_score
    from config import ID2LABEL

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

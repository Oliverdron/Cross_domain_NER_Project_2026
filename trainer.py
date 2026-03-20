"""
trainer.py
----------
Training loop, evaluation, and prediction collection.

Key features (merged from both baselines):
  - Linear warmup + linear decay scheduler  (from multi-file baseline)
  - Gradient clipping at 1.0               (from multi-file baseline)
  - Weight decay via AdamW                 (from multi-file baseline)
  - Best model checkpoint by dev F1        (from both)
  - Per-class seqeval report               (from multi-file baseline)
  - Cross-domain evaluation over all splits (from sam_baseline)
  - Prediction collection for EWT test     (new)
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

from config import ID2LABEL


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _decode_batch(logits, labels) -> tuple:
    """
    Convert raw logits and label tensors to lists of string label sequences.
    Positions with label == -100 (subword continuations / special tokens) are skipped.
    """
    pred_ids  = torch.argmax(logits, dim=-1).cpu().numpy()
    label_ids = labels.cpu().numpy()

    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        preds, golds = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            preds.append(ID2LABEL[int(p)])
            golds.append(ID2LABEL[int(l)])
        true_preds.append(preds)
        true_labels.append(golds)

    return true_preds, true_labels


def evaluate(model, dataloader, device, desc: str = "") -> dict:
    """
    Run inference on a dataloader and return seqeval metrics + full report.
    Also returns raw predictions (list of lists of strings) for saving to file.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}", leave=False):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds, labels = _decode_batch(outputs.logits, batch["labels"])
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / max(len(dataloader), 1)
    f1       = f1_score(all_labels, all_preds)
    report   = classification_report(all_labels, all_preds, digits=4)

    return {
        "loss":        avg_loss,
        "f1":          f1,
        "report":      report,
        "predictions": all_preds,   # raw string predictions — use for saving to file
    }


def train(model, train_loader, dev_loader, device, args) -> str:
    """
    Fine-tune the model on train_loader, selecting the best checkpoint by dev F1.

    Returns the path to the saved best model directory.
    """
    optimizer = AdamW(model.parameters(),
                      lr=args.lr, weight_decay=args.weight_decay)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_f1    = -1.0
    best_model_dir = os.path.join(args.output_dir, "best_model")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader,
                        desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        dev_metrics    = evaluate(model, dev_loader, device, desc="EWT dev")

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss : {avg_train_loss:.4f}")
        print(f"  Dev loss   : {dev_metrics['loss']:.4f}")
        print(f"  Dev F1     : {dev_metrics['f1']:.4f}")
        print(dev_metrics["report"])

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            model.save_pretrained(best_model_dir)
            print(f"  → Best model saved (F1={best_dev_f1:.4f})")

    return best_model_dir

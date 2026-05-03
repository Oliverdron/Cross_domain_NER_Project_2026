import os
import json
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


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


def _decode_batch_full(logits, labels):
    """
    Like _decode_batch but also returns per-token max softmax probability for
    the kept (non-subword, non-special) positions, in the same shape as preds.
    """
    probs     = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_ids  = probs.argmax(axis=-1)
    label_ids = labels.cpu().numpy()

    true_preds, true_labels, true_max_probs = [], [], []
    for prob_seq, pred_seq, label_seq in zip(probs, pred_ids, label_ids):
        preds, golds, mp = [], [], []
        for p_row, p_id, l in zip(prob_seq, pred_seq, label_seq):
            if l == -100:
                continue
            preds.append(ID2LABEL[int(p_id)])
            golds.append(ID2LABEL[int(l)])
            mp.append(float(p_row[int(p_id)]))
        true_preds.append(preds)
        true_labels.append(golds)
        true_max_probs.append(mp)
    return true_preds, true_labels, true_max_probs


def evaluate(model, dataloader, device, desc: str = "", return_full: bool = False) -> dict:
    """
    Run inference on a dataloader and return seqeval metrics + full report.
    Also returns raw predictions (list of lists of strings) for saving to file.

    When return_full=True, additionally return per-token max softmax probability,
    macro/weighted F1, gold/pred entity spans, and a token-level accuracy.
    """
    model.eval()
    all_preds, all_labels = [], []
    all_max_probs = [] if return_full else None
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}", leave=False):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            if return_full:
                preds, labels, mps = _decode_batch_full(outputs.logits, batch["labels"])
                all_max_probs.extend(mps)
            else:
                preds, labels = _decode_batch(outputs.logits, batch["labels"])
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / max(len(dataloader), 1)

    out = {
        "loss":        avg_loss,
        "f1":          f1_score(all_labels, all_preds),
        "precision":   precision_score(all_labels, all_preds),
        "recall":      recall_score(all_labels, all_preds),
        "report":      classification_report(all_labels, all_preds, digits=4),
        "predictions": all_preds,
        "labels":      all_labels,
    }

    if return_full:
        # token-level: flatten and compare directly
        flat_pred  = [p for seq in all_preds  for p in seq]
        flat_gold  = [g for seq in all_labels for g in seq]
        n_tok      = len(flat_pred)
        n_correct  = sum(int(p == g) for p, g in zip(flat_pred, flat_gold))

        gold_spans = [
            [{"type": t, "start": s, "end": e + 1} for (t, s, e) in get_entities(seq)]
            for seq in all_labels
        ]
        pred_spans = [
            [{"type": t, "start": s, "end": e + 1} for (t, s, e) in get_entities(seq)]
            for seq in all_preds
        ]

        out.update({
            "f1_macro":          f1_score(all_labels, all_preds, average="macro"),
            "f1_weighted":       f1_score(all_labels, all_preds, average="weighted"),
            "token_acc":         (n_correct / n_tok) if n_tok else 0.0,
            "per_token_max_prob": all_max_probs,
            "gold_spans":        gold_spans,
            "pred_spans":        pred_spans,
        })
    return out

def train(model, train_loader, dev_loader, device, args,
          early_stopping_patience: int = None,
          train_log_path: str = None,
          best_model_dir: str = None,
          dev_desc: str = "dev") -> dict:
    """
    Fine-tune the model on train_loader, selecting the best checkpoint by dev F1.

    Args:
        early_stopping_patience: stop after N consecutive non-improving dev evals.
                                 None disables early stopping (full args.epochs).
        train_log_path:          if set, write one JSONL line per epoch with
                                 {epoch, train_loss, dev_loss, dev_f1, lr, grad_norm}.
        best_model_dir:          override default save path (defaults to
                                 args.output_dir/best_model).
        dev_desc:                label printed during dev eval (cosmetic).

    Returns:
        dict with keys:
          best_model_dir, best_epoch, best_dev_f1,
          train_time_sec, peak_gpu_mem_mb, epochs_run
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
    best_epoch     = 0
    if best_model_dir is None:
        best_model_dir = os.path.join(args.output_dir, "best_model")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epochs_no_improve = 0
    epochs_run        = 0
    t0                = time.time()

    if train_log_path is not None:
        os.makedirs(os.path.dirname(train_log_path) or ".", exist_ok=True)
        # truncate any pre-existing log
        open(train_log_path, "w").close()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        grad_norm_sum    = 0.0
        grad_norm_count  = 0
        last_lr          = optimizer.param_groups[0]["lr"]

        progress = tqdm(train_loader,
                        desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_sum   += float(grad_norm)
            grad_norm_count += 1
            optimizer.step()
            scheduler.step()
            last_lr = optimizer.param_groups[0]["lr"]

            total_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epochs_run += 1
        avg_train_loss = total_train_loss / len(train_loader)
        dev_metrics    = evaluate(model, dev_loader, device, desc=dev_desc)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss : {avg_train_loss:.4f}")
        print(f"  Dev loss   : {dev_metrics['loss']:.4f}")
        print(f"  Dev F1     : {dev_metrics['f1']:.4f}")
        print(dev_metrics["report"])

        if train_log_path is not None:
            with open(train_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "epoch":      epoch,
                    "train_loss": avg_train_loss,
                    "dev_loss":   dev_metrics["loss"],
                    "dev_f1":     dev_metrics["f1"],
                    "lr":         last_lr,
                    "grad_norm":  (grad_norm_sum / grad_norm_count) if grad_norm_count else 0.0,
                }) + "\n")

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_epoch  = epoch
            model.save_pretrained(best_model_dir)
            print(f"  → Best model saved (F1={best_dev_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"  Early stopping triggered after {epoch} epochs "
                      f"({epochs_no_improve} non-improving).")
                break

    train_time_sec = time.time() - t0
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(device) // (1024 ** 2)
                       if device.type == "cuda" else 0)

    return {
        "best_model_dir":   best_model_dir,
        "best_epoch":       best_epoch,
        "best_dev_f1":      best_dev_f1,
        "train_time_sec":   train_time_sec,
        "peak_gpu_mem_mb":  peak_gpu_mem_mb,
        "epochs_run":       epochs_run,
    }

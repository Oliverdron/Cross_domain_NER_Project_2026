"""Evaluation: span/token/per-type metrics, predictions, confusion matrix.

Span F1 uses seqeval (the same library the baseline uses).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

import sys
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import ID2LABEL, LABEL_LIST  # noqa: E402
from src.data import Example  # noqa: E402


# ── Span helpers ──────────────────────────────────────────────────────────────

def extract_spans(tags: Sequence[str]) -> List[Dict]:
    """Convert a BIO tag sequence into a list of span dicts."""
    spans: List[Dict] = []
    cur_type = None
    cur_start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": i})
            cur_type = tag[2:]
            cur_start = i
        elif tag.startswith("I-"):
            t = tag[2:]
            if cur_type == t:
                continue
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": i})
            cur_type = t
            cur_start = i
        else:
            if cur_type is not None:
                spans.append({"type": cur_type, "start": cur_start, "end": i})
            cur_type = None
            cur_start = None
    if cur_type is not None:
        spans.append({"type": cur_type, "start": cur_start, "end": len(tags)})
    return spans


def collapse_bio(tag: str) -> str:
    if tag == "O" or tag == "-100":
        return "O"
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag[2:]
    return tag


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, dataset, tokenizer, device, batch_size: int) -> Dict:
    """Run model on a tokenized dataset. Returns per-example pred/gold tag
    sequences (only over non-(-100) positions, mapped back to first-subword
    word positions) plus token-level confidence and eval loss.
    """
    collator = DataCollatorForTokenClassification(tokenizer)
    ds_for_loader = dataset.remove_columns(["id"]) if "id" in dataset.column_names else dataset
    loader = DataLoader(ds_for_loader, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    all_pred_tags: List[List[str]] = []
    all_gold_tags: List[List[str]] = []
    all_pred_probs: List[List[float]] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            n_batches += 1
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)
            preds_np = preds.cpu().numpy()
            probs_np = max_probs.cpu().numpy()
            labels_np = batch["labels"].cpu().numpy()
            for p_seq, g_seq, pr_seq in zip(preds_np, labels_np, probs_np):
                p_tags, g_tags, p_conf = [], [], []
                for p, g, pr in zip(p_seq, g_seq, pr_seq):
                    if g == -100:
                        continue
                    p_tags.append(ID2LABEL[int(p)])
                    g_tags.append(ID2LABEL[int(g)])
                    p_conf.append(float(pr))
                all_pred_tags.append(p_tags)
                all_gold_tags.append(g_tags)
                all_pred_probs.append(p_conf)

    return {
        "pred_tags": all_pred_tags,
        "gold_tags": all_gold_tags,
        "pred_probs": all_pred_probs,
        "eval_loss": total_loss / max(n_batches, 1),
    }


# ── Metric computation ────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_per_type(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
                     ) -> Dict[str, Dict[str, float]]:
    """Per-entity-type span P/R/F1 and support."""
    types = sorted({t for g in gold for tag in g if tag != "O" for t in [tag[2:]]} |
                   {t for p in pred for tag in p if tag != "O" for t in [tag[2:]]})
    out: Dict[str, Dict[str, float]] = {}
    for t in types:
        tp = fp = fn = 0
        for g_tags, p_tags in zip(gold, pred):
            g_spans = {(s["start"], s["end"]) for s in extract_spans(g_tags) if s["type"] == t}
            p_spans = {(s["start"], s["end"]) for s in extract_spans(p_tags) if s["type"] == t}
            tp += len(g_spans & p_spans)
            fp += len(p_spans - g_spans)
            fn += len(g_spans - p_spans)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        out[t] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def compute_token_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
                          ) -> Dict[str, float]:
    correct = 0
    total = 0
    tp = fp = fn = 0  # for token-level (any non-O = positive)
    for g_tags, p_tags in zip(gold, pred):
        for g, p in zip(g_tags, p_tags):
            total += 1
            if g == p:
                correct += 1
            g_pos = g != "O"
            p_pos = p != "O"
            if g_pos and p_pos and g == p:
                tp += 1
            elif p_pos and not (g_pos and g == p):
                fp += 1
            elif g_pos and not (p_pos and g == p):
                fn += 1
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {
        "token_acc": _safe_div(correct, total),
        "token_f1": f1,
        "token_precision": prec,
        "token_recall": rec,
    }


def compute_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
                    ) -> Dict:
    span_micro_f1 = float(f1_score(gold, pred, average="micro", zero_division=0))
    span_macro_f1 = float(f1_score(gold, pred, average="macro", zero_division=0))
    span_weighted_f1 = float(f1_score(gold, pred, average="weighted", zero_division=0))
    span_prec = float(precision_score(gold, pred, average="micro", zero_division=0))
    span_rec = float(recall_score(gold, pred, average="micro", zero_division=0))
    per_type = compute_per_type(gold, pred)
    token = compute_token_metrics(gold, pred)
    return {
        "entity_f1_micro": span_micro_f1,
        "entity_f1_macro": span_macro_f1,
        "entity_f1_weighted": span_weighted_f1,
        "entity_precision": span_prec,
        "entity_recall": span_rec,
        **token,
        "per_type": per_type,
        "report": classification_report(gold, pred, digits=4, zero_division=0),
    }


# ── Writers ───────────────────────────────────────────────────────────────────

def write_predictions_jsonl(path: str, examples: Sequence[Example],
                            gold_tags: Sequence[Sequence[str]],
                            pred_tags: Sequence[Sequence[str]],
                            pred_probs: Sequence[Sequence[float]]) -> None:
    """One JSON object per example. Tag sequences are over the *first subword
    of each word* — i.e. word-level after the alignment scheme.

    If truncation occurred, gold/pred sequences are shorter than tokens; we
    align them to the first len(gold) tokens.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex, g, p, pr in zip(examples, gold_tags, pred_tags, pred_probs):
            n = len(g)
            obj = {
                "id": ex.id,
                "tokens": ex.tokens[:n],
                "gold_tags": list(g),
                "pred_tags": list(p),
                "pred_probs": [round(x, 4) for x in pr],
                "gold_spans": extract_spans(g),
                "pred_spans": extract_spans(p),
                "truncated": n < len(ex.tokens),
            }
            f.write(json.dumps(obj) + "\n")


def write_confusion_matrix(path: str, gold_tags: Sequence[Sequence[str]],
                           pred_tags: Sequence[Sequence[str]]) -> None:
    """Token-level, BIO-collapsed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    types = sorted({collapse_bio(t) for seq in list(gold_tags) + list(pred_tags) for t in seq})
    idx = {t: i for i, t in enumerate(types)}
    mat = np.zeros((len(types), len(types)), dtype=np.int64)
    for g_seq, p_seq in zip(gold_tags, pred_tags):
        for g, p in zip(g_seq, p_seq):
            mat[idx[collapse_bio(g)]][idx[collapse_bio(p)]] += 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + types)
        for t, row in zip(types, mat):
            writer.writerow([t] + list(int(x) for x in row))

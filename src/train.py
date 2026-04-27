"""Single (seed, iteration) training run.

Re-initializes the model from the pretrained checkpoint each call (no
warm-starting from prior iterations), trains with early stopping on
target-dev span-F1, returns the best state_dict and training log.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

import sys
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import LABEL2ID, ID2LABEL  # noqa: E402
from src.evaluate import compute_metrics, run_inference  # noqa: E402


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build_model(model_name: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), label2id=LABEL2ID, id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )


@dataclass
class TrainResult:
    best_state_dict: Dict
    best_dev_f1: float
    best_epoch: int
    train_log: List[Dict]
    train_time_sec: float
    peak_gpu_mem_mb: float


def train_one_run(
    train_dataset,
    dev_dataset,
    tokenizer,
    model_name: str,
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    early_stopping_patience: int,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> TrainResult:
    seed_everything(seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = build_model(model_name).to(device)

    collator = DataCollatorForTokenClassification(tokenizer)
    train_for_loader = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]
    )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_for_loader, batch_size=batch_size, shuffle=True,
        collate_fn=collator, generator=g,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state: Optional[Dict] = None
    no_improve = 0
    log: List[Dict] = []
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        grad_norm_sum = 0.0
        n_batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += float(loss.item())
            grad_norm_sum += float(grad_norm)
            n_batches += 1
            progress.set_postfix(loss=f"{float(loss.item()):.4f}")

        avg_train_loss = total_train_loss / max(n_batches, 1)
        avg_grad_norm = grad_norm_sum / max(n_batches, 1)

        dev_inf = run_inference(model, dev_dataset, tokenizer, device, batch_size)
        dev_metrics = compute_metrics(dev_inf["gold_tags"], dev_inf["pred_tags"])
        dev_f1 = dev_metrics["entity_f1_micro"]

        log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_inf["eval_loss"],
            "dev_f1": dev_f1,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": avg_grad_norm,
        })

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break

    train_time = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainResult(
        best_state_dict=best_state,
        best_dev_f1=best_f1,
        best_epoch=best_epoch,
        train_log=log,
        train_time_sec=train_time,
        peak_gpu_mem_mb=peak_mb,
    )

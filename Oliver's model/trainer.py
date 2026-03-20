import os
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report
from torch.optim import AdamW


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def decode_predictions(logits, labels, id2label):
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    label_ids = labels.cpu().numpy()

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(pred_ids, label_ids):
        preds = []
        golds = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            preds.append(id2label[int(p)])
            golds.append(id2label[int(l)])
        true_predictions.append(preds)
        true_labels.append(golds)

    return true_predictions, true_labels


def evaluate(model, dataloader, device, id2label):
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds, golds = decode_predictions(logits, batch["labels"], id2label)
            all_predictions.extend(preds)
            all_labels.extend(golds)

    avg_loss = total_loss / max(len(dataloader), 1)
    f1 = f1_score(all_labels, all_predictions)

    return {
        "loss": avg_loss,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels,
        "report": classification_report(all_labels, all_predictions, digits=4),
    }


def train(model, train_loader, dev_loader, device, args, id2label):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch in progress:
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        dev_metrics = evaluate(model, dev_loader, device, id2label)

        print(f"\nEpoch {epoch}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Dev loss:   {dev_metrics['loss']:.4f}")
        print(f"Dev F1:     {dev_metrics['f1']:.4f}")
        print(dev_metrics["report"])

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    return best_model_path
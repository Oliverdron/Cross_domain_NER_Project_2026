"""
Full per-iteration evaluation bundle for one eval set.

Reuses trainer.evaluate(return_full=True) for span/token metrics and span lists,
adds BIO-collapsed token confusion matrix and per-type P/R/F1/support,
and assembles per-example prediction records for JSONL writing.
"""

from typing import Dict, List, Tuple

from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score as token_f1_score

from trainer import evaluate as trainer_evaluate


def _bio_collapse(label: str) -> str:
    """
    Strip the BIO prefix from a label, merging B- and I- tags into a single type.

    BIO collapsing means that "B-PER" and "I-PER" both become "PER", so a
    token-level confusion matrix groups all tokens of the same entity type
    together regardless of whether they start or continue a span.

    Args:
        label: BIO label string such as "B-PER", "I-LOC", or "O".
    Returns:
        Entity type string ("PER", "LOC", etc.) or "O" for non-entities.
    """
    if label == "O":
        return "O"
    return label.split("-", 1)[1] if "-" in label else label


def collapsed_label_set(labels_seqs: List[List[str]]) -> List[str]:
    """
    Return a sorted list of unique BIO-collapsed entity types seen in labels_seqs.

    Args:
        labels_seqs: list of BIO label sequences (list of lists of strings).
    Returns:
        Sorted list of collapsed type strings (e.g. ["LOC", "O", "PER"]).
    """
    seen = set()
    for seq in labels_seqs:
        for l in seq:
            seen.add(_bio_collapse(l))
    return sorted(seen)


def full_evaluate(model, dataloader, device, eval_set_name: str,
                  example_ids: List[str], example_tokens: List[List[str]]) -> Dict:
    """
    Run evaluate(return_full=True) and assemble all derived artifacts.

    example_ids and example_tokens must align 1:1 with the dataloader's example
    order (i.e. the underlying Dataset must NOT have been shuffled).
    """
    raw = trainer_evaluate(model, dataloader, device, desc=eval_set_name, return_full=True)

    gold_seqs = raw["labels"]
    pred_seqs = raw["predictions"]

    # per-type P/R/F1/support (entity level via seqeval)
    per_type: Dict[str, Dict] = classification_report(
        gold_seqs, pred_seqs, output_dict=True, zero_division=0,
    )

    # token-level confusion (BIO-collapsed)
    flat_gold = [_bio_collapse(l) for seq in gold_seqs for l in seq]
    flat_pred = [_bio_collapse(l) for seq in pred_seqs for l in seq]
    label_order = sorted(set(flat_gold) | set(flat_pred))
    cm = confusion_matrix(flat_gold, flat_pred, labels=label_order)

    # token-level F1 (macro across collapsed labels) — distinct from span F1
    tok_f1 = token_f1_score(flat_gold, flat_pred, labels=label_order,
                            average="macro", zero_division=0)

    # predictions JSONL records
    # The zip below relies on example_ids / example_tokens being in the same order
    # as the dataloader's examples (i.e. the underlying Dataset must NOT have been
    # shuffled — shuffle=False is enforced in _build_eval_loader).
    pred_records: List[Dict] = []
    for i, (ex_id, toks, g_tags, p_tags, g_spans, p_spans, mp) in enumerate(zip(
        example_ids, example_tokens, gold_seqs, pred_seqs,
        raw["gold_spans"], raw["pred_spans"], raw["per_token_max_prob"],
    )):
        pred_records.append({
            "id":                 ex_id,
            "tokens":             toks,
            "gold_tags":          g_tags,
            "pred_tags":          p_tags,
            "gold_spans":         g_spans,
            "pred_spans":         p_spans,
            "per_token_max_prob": mp,
        })

    return {
        "f1":                raw["f1"],
        "f1_macro":          raw.get("f1_macro", 0.0),
        "f1_weighted":       raw.get("f1_weighted", 0.0),
        "precision":         raw["precision"],
        "recall":            raw["recall"],
        "loss":              raw["loss"],
        "token_acc":         raw.get("token_acc", 0.0),
        "token_f1":          tok_f1,
        "report":            raw["report"],
        "per_type":          per_type,
        "confusion_labels":  label_order,
        "confusion_matrix":  cm.tolist(),
        "predictions":       pred_records,
    }

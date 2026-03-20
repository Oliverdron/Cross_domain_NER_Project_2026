import argparse


# EWT (Universal NER) label set — fixed, model always trained against these 7 labels.
# CoNLL MISC → O, WIESP domain types → coarse PER/ORG/LOC (handled in data.py)
LABEL_LIST = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def get_args():
    parser = argparse.ArgumentParser(description="Cross-domain NER baseline")

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",   type=str, default="data_iob2",
                        help="Folder containing all .iob2 files")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Where to save the best model and predictions")

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-cased")

    # ── Training ───────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--max_length",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for linear warmup")

    # ── Misc ───────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

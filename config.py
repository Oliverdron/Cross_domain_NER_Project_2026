import argparse


# EWT (Universal NER) label set — fixed, model always trained against these 7 labels.
# CoNLL MISC → O, WIESP domain types → coarse PER/ORG/LOC (handled in data.py)
LABEL_LIST = [
    # shared / O
    "O",
    # EWT + CoNLL
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    # CoNLL only
    "B-MISC", "I-MISC",
    # WIESP
    "B-Archive", "I-Archive",
    "B-CelestialObject", "I-CelestialObject",
    "B-CelestialObjectRegion", "I-CelestialObjectRegion",
    "B-CelestialRegion", "I-CelestialRegion",
    "B-Citation", "I-Citation",
    "B-Collaboration", "I-Collaboration",
    "B-ComputingFacility", "I-ComputingFacility",
    "B-Database", "I-Database",
    "B-Dataset", "I-Dataset",
    "B-EntityOfFutureInterest", "I-EntityOfFutureInterest",
    "B-Event", "I-Event",
    "B-Fellowship", "I-Fellowship",
    "B-Formula", "I-Formula",
    "B-Grant", "I-Grant",
    "B-Identifier", "I-Identifier",
    "B-Instrument", "I-Instrument",
    "B-Location", "I-Location",
    "B-Mission", "I-Mission",
    "B-Model", "I-Model",
    "B-ObservationalTechniques", "I-ObservationalTechniques",
    "B-Observatory", "I-Observatory",
    "B-Organization", "I-Organization",
    "B-Person", "I-Person",
    "B-Proposal", "I-Proposal",
    "B-Software", "I-Software",
    "B-Survey", "I-Survey",
    "B-Tag", "I-Tag",
    "B-Telescope", "I-Telescope",
    "B-TextGarbage", "I-TextGarbage",
    "B-URL", "I-URL",
    "B-Wavelength", "I-Wavelength",
]
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
    parser.add_argument("--final_eval", action="store_true",
                    help="Also evaluate on test splits. Run only once at the end.")

    # ── Misc ───────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

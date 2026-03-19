from datasets import load_dataset


# ── 1. EWT — local .iob2 files ───────────────────────────────────────────────

EWT_DIR = "./data" 

def parse_iob2(filepath):
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#") or line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split()
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

ewt = {
    "train": parse_iob2(f"{EWT_DIR}/en_ewt-ud-train.iob2"),
    "dev":   parse_iob2(f"{EWT_DIR}/en_ewt-ud-dev.iob2"),
    "test":  parse_iob2(f"{EWT_DIR}/en_ewt-ud-test-masked.iob2"),
}


# ── 2. CoNLL-2003 ─────────────────────────────────────────────────────────────


raw_conll = load_dataset("BramVanroy/conll2003")
conll_label_names = raw_conll["train"].features["ner_tags"].feature.names

def decode_conll(split):
    return [
        {
            "tokens":   ex["tokens"],
            "ner_tags": [conll_label_names[t] for t in ex["ner_tags"]],
        }
        for ex in split
    ]

conll = {
    "train": decode_conll(raw_conll["train"]),
    "dev":   decode_conll(raw_conll["validation"]),  # HF calls it "validation"
    "test":  decode_conll(raw_conll["test"]),
}


# ── 3. WIESP-2022 (astrophysics) ──────────────────────────────────────────────


# WIESP — ner_tags are already strings, just read them directly
raw_wiesp = load_dataset("adsabs/WIESP2022-NER")

def decode_wiesp(split):
    return [
        {
            "tokens":   ex["tokens"],
            "ner_tags": ex["ner_tags"],   
        }
        for ex in split
    ]

wiesp = {
    "train": decode_wiesp(raw_wiesp["train"]),
    "dev":   decode_wiesp(raw_wiesp["validation"]),
    "test":  decode_wiesp(raw_wiesp["test"]),
}


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'Dataset':<12} {'train':>7} {'dev':>7} {'test':>7}")
    print("-" * 36)
    for name, ds in [("EWT", ewt), ("CoNLL-2003", conll), ("WIESP-2022", wiesp)]:
        print(f"{name:<12} {len(ds['train']):>7} {len(ds['dev']):>7} {len(ds['test']):>7}")

    print("\nCoNLL-2003 labels:", conll["train"][0]["ner_tags"][:5], "...")
    print("WIESP-2022 labels:", wiesp["train"][0]["ner_tags"][:5], "...")

    print("\nWIESP label set:", sorted(set(
        t for ex in wiesp["train"] for t in ex["ner_tags"]
    )))
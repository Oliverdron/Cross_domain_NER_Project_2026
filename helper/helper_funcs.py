# helper/helper_funcs.py

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
                parts = line.split("\t")  # your files use tab separation
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def jaccard_vocab(ds_a, ds_b, token_field="tokens"):
    vocab_a = set(t.lower() for ex in ds_a for t in ex[token_field])
    vocab_b = set(t.lower() for ex in ds_b for t in ex[token_field])
    return len(vocab_a & vocab_b) / len(vocab_a | vocab_b)
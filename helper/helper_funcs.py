def parse_iob2(filepath, token_col=0, tag_col=1):
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
                parts = line.split("\t")
                if len(parts) >= max(token_col, tag_col) + 1:
                    tokens.append(parts[token_col])
                    tags.append(parts[tag_col])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

# def jaccard_vocab(ds_a, ds_b, token_field="tokens"):
#     vocab_a = set(t.lower() for ex in ds_a for t in ex[token_field])
#     vocab_b = set(t.lower() for ex in ds_b for t in ex[token_field])
#     return len(vocab_a & vocab_b) / len(vocab_a | vocab_b)


def jaccard_vocab(ds_a, ds_b, token_field="tokens"):
    def get_vocab(ds):
        return set(
            t.lower() for ex in ds 
            for t in ex[token_field]
            if t.isalpha()   # only real words, no punctuation or numbers
        )
    vocab_a = get_vocab(ds_a)
    vocab_b = get_vocab(ds_b)
    return len(vocab_a & vocab_b) / len(vocab_a | vocab_b)
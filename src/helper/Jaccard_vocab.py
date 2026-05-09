def jaccard_vocab(ds_a, ds_b, token_field="tokens"):
    def get_vocab(ds):
        return set(
            t.lower() for ex in ds 
            for t in ex[token_field]
            if t.isalpha()
        )
    vocab_a = get_vocab(ds_a)
    vocab_b = get_vocab(ds_b)
    return len(vocab_a & vocab_b) / len(vocab_a | vocab_b)
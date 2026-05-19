"""
Jaccard_vocab.py — Vocabulary overlap metric between two datasets.

Used in EDA (eda.ipynb) to quantify lexical similarity between source and
target domain corpora via Jaccard index over lowercased alphabetic tokens.
"""


def jaccard_vocab(ds_a: list, ds_b: list, token_field: str = "tokens") -> float:
    """
    Compute the Jaccard index between the token vocabularies of two datasets.

    Only lowercased alphabetic tokens are included (punctuation and numbers skipped).

    Args:
        ds_a:        list of example dicts for the first dataset.
        ds_b:        list of example dicts for the second dataset.
        token_field: key in each example dict that holds the token list.
    Returns:
        |vocab_a ∩ vocab_b| / |vocab_a ∪ vocab_b| in [0.0, 1.0].
    """
    def get_vocab(ds):
        return set(
            t.lower() for ex in ds 
            for t in ex[token_field]
            if t.isalpha()
        )
    vocab_a = get_vocab(ds_a)
    vocab_b = get_vocab(ds_b)
    return len(vocab_a & vocab_b) / len(vocab_a | vocab_b)
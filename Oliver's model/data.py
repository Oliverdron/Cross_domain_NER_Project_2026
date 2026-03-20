import os
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def read_iob2(path: str) -> List[Dict]:
    """
    Reads your .iob2 file.

    Expected format:
    - comment lines start with '#'
    - blank line separates sentences
    - token is column 2 (index 1)
    - NER tag is column 3 (index 2)

    Returns a list of dicts:
    [
        {
            "tokens": [...],
            "labels": [...],
            "lines": [...],   # original token lines
            "comments": [...], # original comment lines before the sentence
        },
        ...
    ]
    """
    sentences = []

    tokens = []
    labels = []
    lines = []
    comments = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                if tokens:
                    sentences.append(
                        {
                            "tokens": tokens,
                            "labels": labels,
                            "lines": lines,
                            "comments": comments,
                        }
                    )
                tokens = []
                labels = []
                lines = []
                comments = []
                continue

            if line.startswith("#"):
                comments.append(line)
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            token = parts[1]
            label = parts[2]

            tokens.append(token)
            labels.append(label)
            lines.append(line)

    if tokens:
        sentences.append(
            {
                "tokens": tokens,
                "labels": labels,
                "lines": lines,
                "comments": comments,
            }
        )

    return sentences


def build_label_vocab(sentences: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({label for sent in sentences for label in sent["labels"]})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


class NERDataset(Dataset):
    def __init__(self, sentences, tokenizer, label2id, max_length=256):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = sent["tokens"]
        labels = sent["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=False,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label2id[labels[word_idx]])
            else:
                # ignore subword continuation pieces in loss
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }
        return item


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def make_dataloaders(train_path, dev_path, test_path, model_name, batch_size, max_length):
    train_sentences = read_iob2(train_path)
    dev_sentences = read_iob2(dev_path)
    test_sentences = read_iob2(test_path)

    label2id, id2label = build_label_vocab(train_sentences)

    tokenizer = get_tokenizer(model_name)

    train_dataset = NERDataset(train_sentences, tokenizer, label2id, max_length=max_length)
    dev_dataset = NERDataset(dev_sentences, tokenizer, label2id, max_length=max_length)
    test_dataset = NERDataset(test_sentences, tokenizer, label2id, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return {
        "train_sentences": train_sentences,
        "dev_sentences": dev_sentences,
        "test_sentences": test_sentences,
        "label2id": label2id,
        "id2label": id2label,
        "tokenizer": tokenizer,
        "train_loader": train_loader,
        "dev_loader": dev_loader,
        "test_loader": test_loader,
    }


def save_label_map(label2id: Dict[str, int], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "labels.txt")
    labels = sorted(label2id.items(), key=lambda x: x[1])
    with open(path, "w", encoding="utf-8") as f:
        for label, idx in labels:
            f.write(f"{idx}\t{label}\n")


def load_label_map(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            idx, label = line.rstrip("\n").split("\t")
            label2id[label] = int(idx)
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
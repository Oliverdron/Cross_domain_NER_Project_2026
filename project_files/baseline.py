import os

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_iob2")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bert_universal_ner")
MODEL_NAME = "bert-base-cased"

PATHS = {
    "universal": {
        "train": os.path.join(DATA_DIR, "universal_train.iob2"),
        "dev": os.path.join(DATA_DIR, "universal_dev.iob2"),
        "test": os.path.join(DATA_DIR, "universal_test_masked.iob2"),
    },
    "news": {
        "train": os.path.join(DATA_DIR, "news_train.iob2"),
        "dev": os.path.join(DATA_DIR, "news_dev.iob2"),
        "test": os.path.join(DATA_DIR, "news_test.iob2"),
    },
    "astro": {
        "train": os.path.join(DATA_DIR, "astro_train.iob2"),
        "dev": os.path.join(DATA_DIR, "astro_dev.iob2"),
        "test": os.path.join(DATA_DIR, "astro_test.iob2"),
    }
}

TRAIN_FILE = PATHS["universal"]["train"]
DEV_FILE = PATHS["universal"]["dev"]


def read_universal_iob2(path):
    sentences = []
    tokens = []
    ner_tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens = []
                    ner_tags = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[1]

            tokens.append(token)
            ner_tags.append(tag)

    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": ner_tags})

    return sentences


def build_label_mappings(train_data, dev_data):
    labels = sorted(
        set(tag for ex in train_data for tag in ex["ner_tags"]) |
        set(tag for ex in dev_data for tag in ex["ner_tags"])
    )
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return labels, label2id, id2label


def convert_tags_to_ids(data, label2id):
    converted = []
    for ex in data:
        converted.append({
            "tokens": ex["tokens"],
            "ner_tags": [label2id[tag] for tag in ex["ner_tags"]],
        })
    return converted


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )

    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(examples["ner_tags"][i][word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics_factory(id2label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            curr_preds = []
            curr_labels = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                curr_preds.append(id2label[int(pred_id)])
                curr_labels.append(id2label[int(label_id)])
            true_predictions.append(curr_preds)
            true_labels.append(curr_labels)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading data...")
    train_data = read_universal_iob2(TRAIN_FILE)
    dev_data = read_universal_iob2(DEV_FILE)

    print(f"Train sentences: {len(train_data)}")
    print(f"Dev sentences:   {len(dev_data)}")

    labels, label2id, id2label = build_label_mappings(train_data, dev_data)
    print("Labels:", labels)

    train_data = convert_tags_to_ids(train_data, label2id)
    dev_data = convert_tags_to_ids(dev_data, label2id)

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "dev": Dataset.from_list(dev_data),
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(id2label),
    )

    print("Training...")
    trainer.train()

    print("Evaluating on dev...")
    metrics = trainer.evaluate()
    print(metrics)

    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"Saved best model to {best_model_dir}")


if __name__ == "__main__":
    main()
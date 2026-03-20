import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import read_iob2, load_label_map
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def predict_sentence(model, tokenizer, tokens, id2label, max_length, device):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    word_predictions = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_predictions:
            word_predictions[word_idx] = id2label[pred_ids[token_idx]]

    preds = [word_predictions[i] for i in range(len(tokens))]
    return preds


def write_predictions(sentences, pred_sequences, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for sent, preds in zip(sentences, pred_sequences):
            for c in sent["comments"]:
                f.write(c + "\n")

            for line, pred in zip(sent["lines"], preds):
                parts = line.split("\t")
                if len(parts) < 3:
                    f.write(line + "\n")
                    continue
                parts[2] = pred
                f.write("\t".join(parts) + "\n")

            f.write("\n")


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    label_map_path = os.path.join(args.model_dir, "labels.txt")
    model_path = os.path.join(args.model_dir, "best_model.pt")

    label2id, id2label = load_label_map(label_map_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = build_model(args.model_name, label2id, id2label)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    sentences = read_iob2(args.input_path)

    all_preds = []
    for sent in tqdm(sentences, desc="Predicting"):
        preds = predict_sentence(
            model=model,
            tokenizer=tokenizer,
            tokens=sent["tokens"],
            id2label=id2label,
            max_length=args.max_length,
            device=device,
        )
        all_preds.append(preds)

    write_predictions(sentences, all_preds, args.output_path)
    print(f"Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
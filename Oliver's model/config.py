import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()
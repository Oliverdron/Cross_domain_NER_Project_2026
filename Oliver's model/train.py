import os
import torch

from config import get_args
from data import make_dataloaders, save_label_map
from model import build_model
from trainer import set_seed, train, evaluate


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    set_seed(args.seed)

    data_bundle = make_dataloaders(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    save_label_map(data_bundle["label2id"], args.output_dir)

    model = build_model(
        model_name=args.model_name,
        label2id=data_bundle["label2id"],
        id2label=data_bundle["id2label"],
    )
    model.to(device)

    best_model_path = train(
        model=model,
        train_loader=data_bundle["train_loader"],
        dev_loader=data_bundle["dev_loader"],
        device=device,
        args=args,
        id2label=data_bundle["id2label"],
    )

    print(f"\nLoading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(
        model=model,
        dataloader=data_bundle["test_loader"],
        device=device,
        id2label=data_bundle["id2label"],
    )

    print("\nFinal test results")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test F1:   {test_metrics['f1']:.4f}")
    print(test_metrics["report"])


if __name__ == "__main__":
    main()
# Cross-Domain NER with BERT

This project implements a cross-domain Named Entity Recognition (NER) system using BERT. It fine-tunes a BERT model on the EWT (English Web Treebank) dataset and evaluates its performance on three different domains: EWT (in-domain), CoNLL-2003 (similar), and WIESP-2022 (different).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Cross_domain_NER_Project_2026
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point is `main.py`. You can run the training and evaluation pipeline with default settings:

```bash
python main.py
```

### Customizing Training

You can customize the training parameters using command-line arguments:

```bash
python main.py \
  --model_name_or_path bert-base-uncased \
  --train_dataset ewt \
  --dev_dataset conll2003 \
  --test_dataset wiesp \
  --output_dir ./outputs \
  --epochs 3 \
  --lr 2e-5 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1
```

**Available Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name_or_path` | Pre-trained model to use | `"bert-base-uncased"` |
| `--train_dataset` | Training dataset | `"ewt"` |
| `--dev_dataset` | Development dataset | `"conll2003"` |
| `--test_dataset` | Test dataset | `"wiesp"` |
| `--output_dir` | Output directory for checkpoints and predictions | `./outputs` |
| `--epochs` | Number of training epochs | `5` |
| `--lr` | Learning rate | `2e-5` |
| `--batch_size` | Batch size | `16` |
| `--weight_decay` | Weight decay | `0.01` |
| `--warmup_ratio` | Warmup ratio for learning rate scheduler | `0.1` |

## Output

The script will:
1.  Train the model on the specified training dataset.
2.  Evaluate on the development dataset after each epoch.
3.  Save the best model (by dev F1) to `./outputs/best_model`.
4.  Evaluate the best model on all three datasets (EWT, CoNLL-2003, WIESP-2022).
5.  Save predictions for the EWT test set to `./outputs/ewt_test_predictions.iob2`.

## Datasets

The project uses the following datasets:

-   **EWT (English Web Treebank):** In-domain training data.
-   **CoNLL-2003:** Similar domain, used for development.
-   **WIESP-2022:** Different domain, used for cross-domain evaluation.

## License

This project is for educational purposes as part of the Natural Language and Deep Learning course at ITU.

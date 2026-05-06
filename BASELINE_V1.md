# Cross-Domain NER Baseline

                            ...

## File structure

```
├── main.py         ← entry point, run this
├── config.py       ← hyperparameters and label set
├── data.py         ← parsing, tokenisation, label normalisation, prediction saving
├── model.py        ← builds the BERT token classification model
├── trainer.py      ← training loop, evaluation, best-model checkpointing
└── data/      ← local .iob2 files
    ├── universal_train.iob2
    ├── universal_dev.iob2
    ├── universal_test_masked.iob2
    ├── news_train.iob2
    ├── news_dev.iob2
    ├── news_test.iob2
    ├── astro_train.iob2
    ├── astro_dev.iob2
    └── astro_test.iob2
```

## Installation

```bash
pip install transformers datasets torch seqeval evaluate
```

## Usage

**Default run (trains for 5 epochs on EWT, evaluates on dev splits):**
```bash
  python main.py [--data_dir data] [--output_dir outputs] [--epochs 5] ...
```
Full argument list: see config.py or run  python main.py --help

**With custom settings:**
```bash
python main.py \
  --data_dir data \
  --output_dir outputs \
  --model_name google-bert/bert-base-cased \
  --epochs 5 \
  --batch_size 16 \
  --max_length 256 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --seed 42 \
  --device cuda
```

**Final evaluation (also runs on test splits — only once at the end):**
```bash
python main.py --final_eval
```

**CPU only:**
```bash
python main.py --device cpu
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Folder containing all .iob2 files | `data` |
| `--output_dir` | Where to save the best model and predictions | `outputs` |
| `--model_name` | Pre-trained model name | `google-bert/bert-base-cased` |
| `--epochs` | Number of training epochs | `5` |
| `--batch_size` | Batch size | `16` |
| `--max_length` | Max token sequence length | `256` |
| `--lr` | Learning rate | `5e-5` |
| `--weight_decay` | Weight decay for AdamW | `0.01` |
| `--warmup_ratio` | Fraction of steps used for linear warmup | `0.1` |
| `--seed` | Random seed | `42` |
| `--device` | Device to use (`cuda` or `cpu`) | `cuda` |
| `--final_eval` | Also evaluate on test splits (run once at the end) | `False` |

## Output

| File | Description |
|------|-------------|
| `outputs/best_model/` | Best BERT checkpoint selected by EWT dev F1 |
| `outputs/ewt_test_predictions.iob2` | EWT test predictions — upload to LearnIT for official scoring |

## Datasets

All three datasets must be placed in `data/` before running.

- **EWT** (`universal_*.iob2`) — English web text (blogs, forums, emails). Source domain. 5-column IOB2 format.
- **CoNLL-2003** (`news_*.iob2`) — Reuters newswire. Similar domain. 2-column IOB2 format.
- **WIESP-2022** (`astro_*.iob2`) — Astrophysics paper abstracts. Different domain. 2-column IOB2 format.

## Label set

The model uses a unified label set combining all three datasets (65 labels). Overlapping labels are merged: WIESP `Person` → `PER`, `Organization` → `ORG`, `Location` → `LOC`. CoNLL `MISC` is mapped to `O` as it has no equivalent in the other datasets.
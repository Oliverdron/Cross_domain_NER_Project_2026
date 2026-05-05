# Architecture

**Analysis Date:** 2026-05-05

## Summary

This project implements cross-domain Named Entity Recognition (NER) using BERT-based token classification. The core research question is how iteratively injecting target-domain training examples into a source-domain fine-tuned model affects performance across three NER domains (EWT/universal, CoNLL-2003/news, WIESP-2022/astro). Two execution modes exist: a baseline single-train script (`baseline_main_v1.py`) and a full iterative experiment runner (`scripts/run_experiment.py`).

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Entry Points                                 │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  baseline_main_v1.py │ scripts/run_          │ scripts/run_          │
│  (single EWT→all     │ experiment.py         │ baselines.py          │
│   cross-eval)        │ (iterative injection) │ (single-domain baseln)│
└──────────┬───────────┴──────────┬────────────┴──────────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Core Pipeline Layer                              │
│   data.py          model.py        trainer.py       config.py        │
│   (parse/tokenise/ (build_model    (train/evaluate/ (label schema +  │
│    collate)         from HF)        metrics)         argparse)        │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  src/experiment/ (iterative infra)                   │
│  config_loader.py  injection.py  evaluation.py  logging_io.py        │
│  (YAML→frozen      (pool build/  (full_evaluate (atomic file I/O,    │
│   dataclasses)      slice)        + confusion)   summary.csv)         │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Pretrained HF Model (google-bert/bert-base-cased by default)        │
│  AutoModelForTokenClassification — fine-tuned each iteration/run     │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Output Artifacts                                                    │
│  runs/<exp_name>/summary.csv                                         │
│  runs/<exp_name>/seeds/seed_<N>/iter_<NNN>/                          │
│    metrics.json  meta.json  per_type_metrics.json                    │
│    confusion_matrix_<set>.csv  predictions_<set>.jsonl               │
│    train_log.jsonl                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `data.parse_iob2` | Parse `.iob2` files into list-of-dict examples with `id`, `tokens`, `ner_tags`, `raw_lines`, `unit` | `data.py:40` |
| `data.normalize_tag` | Unify heterogeneous labels across datasets (WIESP→PER/ORG/LOC, MISC→O) | `data.py:23` |
| `data.make_tokenize_fn` | Subword tokenize + align NER labels (-100 for non-first subwords and special tokens) | `data.py:173` |
| `data.prepare_split` | Convert raw example list to a HuggingFace `Dataset` ready for DataLoader | `data.py:222` |
| `data.make_dataloader` | Wrap `Dataset` in `DataLoader` with `DataCollatorForTokenClassification` | `data.py:249` |
| `data.save_predictions` | Write model predictions back to `.iob2` format | `data.py:258` |
| `model.build_model` | Load `AutoModelForTokenClassification` from HF with unified label schema | `model.py:12` |
| `trainer.train` | Fine-tune with AdamW + linear warmup + early stopping; save best checkpoint by dev F1 | `trainer.py:134` |
| `trainer.evaluate` | Inference loop; returns seqeval F1/P/R + optional span lists, token probs | `trainer.py:70` |
| `config.LABEL_LIST` | Single source of truth for the 57-label unified NER schema | `config.py:4` |
| `config.get_args` | CLI argparse for baseline training (model, data paths, hyperparams) | `config.py:45` |
| `src/experiment/config_loader.py` | Parse YAML experiment config into frozen dataclasses; collect dataset SHA-256 hashes | `src/experiment/config_loader.py` |
| `src/experiment/injection.py` | Build seed-deterministic injection pool; slice first `k * step_size` examples for iteration `k` | `src/experiment/injection.py` |
| `src/experiment/evaluation.py` | Extend `trainer.evaluate` with confusion matrix, per-type F1, and per-example JSONL records | `src/experiment/evaluation.py` |
| `src/experiment/logging_io.py` | Atomic file writes (JSON, JSONL, CSV); append-only `summary.csv`; per-iteration directory init | `src/experiment/logging_io.py` |
| `src/helper/helper_funcs.py` | Standalone IOB2 parser (simpler version) and Jaccard vocabulary overlap utility | `src/helper/helper_funcs.py` |
| `baseline_main_v1.py` | Train-once on EWT, cross-eval on CoNLL/WIESP dev (and optionally test) | `baseline_main_v1.py` |
| `scripts/run_experiment.py` | Iterative injection loop: seeds × iterations; loads YAML config | `scripts/run_experiment.py` |
| `scripts/run_baselines.py` | Three single-domain baselines (ewt_only, conll_only, astro_only) matching the iterative output schema | `scripts/run_baselines.py` |

## NER Label Schema

Defined in `config.py:LABEL_LIST` — 57 labels total in IOB2 format:
- **Universal (all 3 datasets):** `O`, `B/I-PER`, `B/I-ORG`, `B/I-LOC`
- **WIESP/Astro-specific (24 types):** `B/I-CelestialObject`, `B/I-Telescope`, `B/I-Mission`, `B/I-Software`, etc.
- **Normalization rules** (in `data.normalize_tag`): WIESP `Person`→`PER`, `Organization`→`ORG`, `Location`→`LOC`; CoNLL `MISC`→`O`

## Data Flow

### Baseline Training Pipeline (`baseline_main_v1.py`)

1. Parse CLI args via `config.get_args()` (`config.py:45`)
2. Load all three datasets: `data.load_all_datasets(data_dir)` (`data.py:118`)
3. Tokenize splits: `data.prepare_split(sentences, tokenizer, max_length)` (`data.py:222`)
4. Build DataLoaders: `data.make_dataloader(dataset, tokenizer, batch_size)` (`data.py:249`)
5. Build fresh model: `model.build_model(args.model_name)` (`model.py:12`)
6. Fine-tune on EWT train: `trainer.train(model, train_loader, ewt_dev_loader, device, args)` (`trainer.py:134`)
7. Reload best checkpoint from `outputs/best_model/`
8. Evaluate on EWT dev, CoNLL dev/test, WIESP dev/test: `trainer.evaluate(model, loader, device)` (`trainer.py:70`)
9. Save EWT test predictions: `data.save_predictions(sentences, predictions, path)` (`data.py:258`)

### Iterative Injection Pipeline (`scripts/run_experiment.py`)

1. Load YAML config → `ExperimentConfig` frozen dataclass (`src/experiment/config_loader.py:72`)
2. Parse source train + target train/dev + named eval sets from `.iob2` files
3. Tokenize eval splits and target dev ONCE (these do not change across iterations)
4. For each seed:
   a. Build deterministic injection pool: `injection.build_injection_pool(target_train, seed, out_dir)` → writes `injection_order.json`
   b. For iteration `k` in `0..n_iterations`:
      - Slice pool: first `k * step_size` examples → `target_chunk`
      - Build training mix: `source_train + target_chunk`
      - Tokenize mix (fresh each iter; no cache to keep truncation counter accurate)
      - Instantiate FRESH model from pretrained checkpoint (never carries weights across iterations)
      - Train with early stopping on target dev F1
      - Reload best checkpoint; delete checkpoint dir after loading
      - Evaluate on all eval sets via `src/experiment/evaluation.full_evaluate`
      - Persist: `meta.json`, `metrics.json`, `per_type_metrics.json`, `confusion_matrix_<set>.csv`, `predictions_<set>.jsonl`, `train_log.jsonl`, `added_target_ids.txt`
      - Append row to `summary.csv`

### Training Loop Details (`trainer.py:train`)

- Optimizer: AdamW with `lr`, `weight_decay`
- Scheduler: linear warmup for `warmup_ratio * total_steps`, then linear decay to 0
- Gradient clipping: max norm 1.0 (`torch.nn.utils.clip_grad_norm_`)
- Best checkpoint: saved via `model.save_pretrained(best_model_dir)` when dev F1 improves
- Early stopping: `early_stopping_patience` consecutive non-improving dev epochs
- Training log: one JSONL line per epoch → `train_log.jsonl`
- Returns: `{best_model_dir, best_epoch, best_dev_f1, train_time_sec, peak_gpu_mem_mb, epochs_run}`

### Label Alignment in Tokenization

`data.make_tokenize_fn` (`data.py:173`) handles subword alignment:
- `[CLS]`, `[SEP]`, padding positions → label `-100` (ignored by cross-entropy loss)
- Subword continuation tokens → label `-100`
- First subword of each word → `LABEL2ID[normalize_tag(raw_tag)]`

## Datasets

| Key | Files | Format | Token col | Tag col | Unit |
|-----|-------|--------|-----------|---------|------|
| `ewt` | `universal_{train,dev,test_masked}.iob2` | 5-column TSV | 1 | 2 | sentence |
| `conll` | `news_{train,dev,test}.iob2` | 2-column TSV | 0 | 1 | sentence |
| `wiesp`/`astro` | `astro_{train,dev,test}.iob2` | 2-column TSV | 0 | 1 | paragraph |

EWT test gold tags are masked — predictions are saved to `outputs/ewt_test_predictions.iob2` for external LearnIT evaluation.

## Experiment Configurations

Two YAML configs drive the iterative experiments:

- `experiments/config_conll.yaml`: source=EWT → target=CoNLL; 150 sentences/step; 8 iterations; max_seq_len=512; batch=32
- `experiments/config_astro.yaml`: source=EWT → target=Astro (paragraphs); 5 paragraphs/step; 8 iterations; max_seq_len=512; batch=32

Both use seeds `[42, 123, 456]`, up to 300 epochs with `early_stopping_patience=8`.

## Error Handling

**Strategy:** No custom exception hierarchy. Errors propagate as Python exceptions. The `os.replace` atomic-write pattern in `logging_io.py` prevents partial output files on crash.

**Truncation tracking:** `prepare_split(..., return_truncation_count=True)` counts examples where tokenization cut off words. Warnings printed for astro (paragraphs) if >10% truncated.

**Device fallback:** Both entry points fall back from `cuda` to `cpu` if CUDA is unavailable.

## Architectural Constraints

- **Threading:** Single-threaded training; DataLoader uses default worker count (no explicit `num_workers`)
- **Global state:** `config.LABEL_LIST`, `config.LABEL2ID`, `config.ID2LABEL` are module-level constants imported by `data.py`, `trainer.py`, and `logging_io.py`
- **Model re-init:** The iterative runner explicitly re-initializes the model from the pretrained HF checkpoint at every iteration — do not move `build_model()` outside the loop in `scripts/run_experiment.py:224`
- **Checkpoint lifecycle:** Best checkpoint dir is created by `trainer.train`, loaded by caller, then deleted with `shutil.rmtree`. The directory exists only transiently during a run.
- **HF caching:** On HPC, `HF_HOME` is set to `/home/olgy/.cache/huggingface` so the model is fetched once across jobs

## Anti-Patterns

### Two separate IOB2 parsers

**What happens:** `src/helper/helper_funcs.py` contains an older, simpler `parse_iob2` function that lacks `id`, `raw_lines`, `unit` fields. `data.py` has the canonical implementation.

**Why it's wrong:** Code duplication. Callers using `helper_funcs.parse_iob2` will get examples missing `id` and `raw_lines`, which breaks `injection.py` and `logging_io.py`.

**Do this instead:** Always use `data.parse_iob2`. The helper version (`src/helper/helper_funcs.py`) should be treated as dead code or removed.

### Hardcoded output path in baseline_main_v1.py

**What happens:** `baseline_main_v1.py` saves the best model to `outputs/best_model` and predictions to `outputs/ewt_test_predictions.iob2`. The iterative runner uses per-iteration dirs under `runs/`.

**Why it's wrong:** The baseline and iterative outputs use incompatible directory conventions, and the baseline does not write `summary.csv`, making direct comparison harder.

**Do this instead:** Use `scripts/run_baselines.py` for proper comparable baselines; `baseline_main_v1.py` is for quick one-shot experiments only.

## Cross-Cutting Concerns

**Logging:** `print()` to stdout; structured per-epoch logging to `train_log.jsonl`; no logging framework.

**Validation:** Input validation limited to IOB2 parser (column bounds check) and `unit` parameter check. Config validation via YAML schema + Python dataclass construction (missing required keys raise `TypeError`).

**Reproducibility:** `trainer.set_seed()` sets Python/NumPy/PyTorch/CUDA seeds. Injection order written to `injection_order.json` with SHA-256 hash for audit. Config snapshot with git commit hash + dataset SHA-256 written at experiment start.

---

*Architecture analysis: 2026-05-05*

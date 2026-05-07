<!-- refreshed: 2026-05-07 -->
# Architecture

**Analysis Date:** 2026-05-07

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Points (CLI)                            │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  baseline_main_v1.py │ scripts/run_baselines│ scripts/run_experiment│
│  (single train+eval) │   .py (3 baselines)  │  .py (iterative mix)  │
└──────────┬───────────┴──────────┬───────────┴──────────┬────────────┘
           │                      │                      │
           │       ┌──────────────┴──────────────────────┘
           │       │      `experiments/*.yaml` (YAML config)
           │       │              │
           │       │              ▼
           │       │      ┌──────────────────────────────────────┐
           │       │      │   src/experiment/config_loader.py    │
           │       │      │   (frozen dataclasses + hashes)      │
           │       │      └──────────────────────────────────────┘
           ▼       ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                       Core Pipeline (top-level)                      │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐ │
│  │  config.py  │   │   data.py    │   │  model.py  │   │trainer.py│ │
│  │ argparse +  │──▶│ parse_iob2 / │──▶│build_model │──▶│ train/   │ │
│  │ LABEL_LIST  │   │ tokenize +   │   │ (BERT TC)  │   │ evaluate │ │
│  │   (65)      │   │ DataLoader   │   │            │   │ + ckpt   │ │
│  └─────────────┘   └──────────────┘   └────────────┘   └──────────┘ │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Experiment Helpers (src/experiment/)                │
│  injection.py (target pool) · evaluation.py (full metrics) ·         │
│  logging_io.py (atomic CSV/JSON/JSONL writers)                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Outputs:  runs/<exp>/seeds/seed_<S>/iter_<NNN>/  +  summary.csv     │
│  Inputs :  data/*.iob2  (EWT, CoNLL-2003, WIESP-2022)                │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Argument / label registry | CLI args + unified 65-label set + LABEL2ID/ID2LABEL | `config.py` |
| Data layer | IOB2 parsing, label normalisation, tokenisation/alignment, DataLoader, prediction writing | `data.py` |
| Model factory | Build `AutoModelForTokenClassification` with project's label maps | `model.py` |
| Training engine | Seeding, AdamW + linear warmup, train loop, dev evaluation, best-checkpoint save, JSONL train log | `trainer.py` |
| Baseline driver | Default single-domain train + cross-domain eval (`final_eval`), saves EWT predictions | `baseline_main_v1.py` |
| Iterative driver | Source+target mix per iteration k, fresh model each iter, full metrics + summary.csv | `scripts/run_experiment.py` |
| Per-domain baselines | EWT-only / CoNLL-only / Astro-only → same summary schema for plotting overlay | `scripts/run_baselines.py` |
| Experiment config | Frozen dataclasses for YAML configs; dataset SHA256 + git commit capture | `src/experiment/config_loader.py` |
| Target injection | Seeded shuffle of target pool, deterministic slicing per iteration | `src/experiment/injection.py` |
| Full evaluation | seqeval + sklearn confusion + per-type metrics + per-token max-prob | `src/experiment/evaluation.py` |
| Atomic IO | Tmp+rename writers for JSON/JSONL/CSV, summary row schema, run dir layout | `src/experiment/logging_io.py` |
| Legacy helpers | Earlier minimal `parse_iob2` + Jaccard vocab overlap (used by EDA only) | `src/helper/helper_funcs.py` |
| HPC orchestration | SLURM job scripts for env install, smoke test, real training | `hpc/install_env.job`, `hpc/smoke_test.job`, `hpc/train_iter.job` |

## Pattern Overview

**Overall:** Layered ML pipeline (data → model → trainer) with two parallel
top-level entry-point variants and a thin `src/experiment/` toolkit for the
iterative cross-domain protocol.

**Key Characteristics:**
- Hugging Face `transformers` + `datasets` over PyTorch `DataLoader`; no Trainer API — a hand-written training loop in `trainer.py:train`
- Single unified label space (65 labels in `config.py:LABEL_LIST`) shared across EWT/CoNLL/WIESP via `data.py:UNIFY_MAP` + `normalize_tag`
- All experiment configuration is data: top-level CLI args (`config.py:get_args`) for the baseline, YAML + frozen dataclass (`src/experiment/config_loader.py:ExperimentConfig`) for iterative runs
- Reproducibility: seed everywhere (`trainer.set_seed`), per-run `config.json` snapshot with dataset SHA256 + git commit (`src/experiment/logging_io.py:write_config_snapshot`)
- Atomic file IO everywhere: tmp file + `os.replace` (`src/experiment/logging_io.py:_atomic_write`)
- Fresh model per iteration in iterative runs — re-loaded from pretrained checkpoint, never carried over (`scripts/run_experiment.py:224`, marked with explicit comment)

## Layers

**Configuration layer:**
- Purpose: Centralise hyperparameters, label set, and per-experiment YAML
- Location: `config.py`, `experiments/*.yaml`, `src/experiment/config_loader.py`
- Contains: `LABEL_LIST`, `LABEL2ID`, `ID2LABEL`, `get_args()`, `ExperimentConfig` (frozen dataclass), `load_config`, `collect_dataset_hashes`, `git_commit_hash`
- Depends on: `argparse`, `pyyaml`, `dataclasses`, `hashlib`, `subprocess`
- Used by: every entry point and every other layer (label maps), `src/experiment/logging_io.py` (LABEL_LIST → ENTITY_TYPES schema)

**Data layer:**
- Purpose: Parse IOB2 files, normalise tags, tokenise with sub-word alignment, build DataLoaders, write predictions back to IOB2
- Location: `data.py`
- Contains: `parse_iob2`, `normalize_tag`, `entity_density`, `load_all_datasets`, `make_tokenize_fn`, `prepare_split`, `make_dataloader`, `save_predictions`
- Depends on: `transformers.AutoTokenizer`, `transformers.DataCollatorForTokenClassification`, `datasets.Dataset`, `config.LABEL2ID/ID2LABEL/LABEL_LIST`
- Used by: `baseline_main_v1.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`

**Model layer:**
- Purpose: Wrap `AutoModelForTokenClassification.from_pretrained` with project label maps
- Location: `model.py` (intentionally minimal — see file docstring)
- Contains: `build_model(model_name)`
- Depends on: `transformers.AutoModelForTokenClassification`, `config.LABEL2ID`, `config.ID2LABEL`
- Used by: `baseline_main_v1.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`

**Training layer:**
- Purpose: Hand-rolled fine-tuning loop; per-epoch dev eval and best-checkpoint selection by dev F1
- Location: `trainer.py`
- Contains: `set_seed`, `_decode_batch`, `_decode_batch_full`, `evaluate(model, dataloader, device, return_full=…)`, `train(...)`
- Depends on: `torch`, `torch.optim.AdamW`, `transformers.get_linear_schedule_with_warmup`, `seqeval.metrics`, `tqdm`
- Used by: `baseline_main_v1.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`, `src/experiment/evaluation.py` (re-exports `evaluate` with `return_full=True`)

**Experiment-helper layer (`src/experiment/`):**
- Purpose: Iterative-experiment-specific tooling on top of the core pipeline
- Location: `src/experiment/`
- Contains:
  - `config_loader.py` — YAML → frozen dataclass + dataset hashing + git commit
  - `injection.py` — `build_injection_pool` (seeded shuffle, persisted), `slice_for_iter`, `select_examples`
  - `evaluation.py` — `full_evaluate` (calls `trainer.evaluate(return_full=True)`, adds confusion matrix + per-type stats + JSONL records)
  - `logging_io.py` — atomic JSON/JSONL/CSV writers, `SUMMARY_COLUMNS` schema, `init_iter_dir`, `init_seed_dir`, `run_root`
- Depends on: `trainer.evaluate`, `config.LABEL_LIST`, `seqeval`, `sklearn`, `pyyaml`
- Used by: `scripts/run_experiment.py` and `scripts/run_baselines.py`

**Orchestration / HPC layer:**
- Purpose: SLURM job scripts for ITU HPC, plus standalone runners
- Location: `scripts/`, `hpc/`
- Contains: `run_experiment.py`, `run_baselines.py`, `train_iter.job`, `smoke_test.job`, `install_env.job`
- Depends on: All layers above
- Used by: humans (CLI / `sbatch`)

## Data Flow

### Baseline path (`python baseline_main_v1.py`)

1. CLI args parsed (`config.py:get_args`) and seed set (`trainer.set_seed`) (`baseline_main_v1.py:29-39`)
2. All three datasets loaded as lists of dicts (`data.load_all_datasets` → `data.parse_iob2`) (`baseline_main_v1.py:43`, `data.py:118-168`)
3. Each split tokenised + label-aligned via `data.prepare_split` → `datasets.Dataset` with `input_ids`, `attention_mask`, `labels` columns (`baseline_main_v1.py:48-54`, `data.py:222-245`)
4. `DataLoader`s built with `DataCollatorForTokenClassification` (dynamic padding) (`data.py:248-252`)
5. `model.build_model` returns a `BertForTokenClassification` head with 65 outputs (`model.py:12-19`)
6. `trainer.train` runs AdamW + linear warmup, evaluates dev each epoch, saves best to `<output_dir>/best_model/` (`trainer.py:134-258`)
7. Best checkpoint reloaded via `AutoModelForTokenClassification.from_pretrained` (`baseline_main_v1.py:81`)
8. Cross-domain eval over EWT-dev, CoNLL-dev, WIESP-dev (and tests if `--final_eval`) using `trainer.evaluate` (`baseline_main_v1.py:101-105`)
9. EWT test predictions written via `data.save_predictions` to `<output_dir>/ewt_test_predictions.iob2` (`baseline_main_v1.py:109-116`, `data.py:257-273`)

### Iterative experiment path (`python scripts/run_experiment.py --config experiments/config_*.yaml`)

1. YAML loaded → `ExperimentConfig` frozen dataclass; dataset SHA256 + git commit captured (`scripts/run_experiment.py:112`, `src/experiment/config_loader.py:69-129`)
2. Source/target/eval splits parsed via `data.parse_iob2` with per-dataset column rules from `_IOB2_COLS` (`scripts/run_experiment.py:63-76`)
3. Eval splits + target dev tokenised **once** and cached as DataLoaders (`scripts/run_experiment.py:160-174`)
4. Run dir created at `runs/<experiment_name>/`; config snapshot written (`scripts/run_experiment.py:177-187`)
5. For each `seed` in config:
   - Target injection pool built and persisted to `seeds/seed_<S>/injection_order.json` (`src/experiment/injection.py:16-39`)
   - For each iteration `k` with `n_target = schedule[k]`:
     - First `n_target` ids selected from pool → mixed with full source train (`scripts/run_experiment.py:207-209`)
     - Mix tokenised → fresh `train_loader`
     - **Fresh model rebuilt from pretrained checkpoint** (`scripts/run_experiment.py:224`, comment "DO NOT MOVE OUTSIDE LOOP")
     - `trainer.train` runs with early stopping (`trainer.py:240-245`); per-epoch JSONL log → `iter_dir/train_log.jsonl`
     - Best checkpoint reloaded then deleted (`shutil.rmtree`) — only metrics persist (`scripts/run_experiment.py:241-244`)
     - Each eval set scored via `src/experiment/evaluation.py:full_evaluate` → metrics + per-type + confusion + per-token max-prob
     - Outputs persisted atomically via `logging_io`: `meta.json`, `metrics.json`, `per_type_metrics.json`, `confusion_matrix_<name>.csv`, `predictions_<name>.jsonl`, `added_target_ids.txt`
     - One row per eval set appended to `runs/<exp>/summary.csv` (shared across seeds and iterations)

### Per-domain baseline path (`python scripts/run_baselines.py`)

1. Three baselines (`ewt_only`, `conll_only`, `astro_only`) defined in `BASELINES` dict at module top (`scripts/run_baselines.py:56-84`)
2. For each baseline × seed: train once on its own data with early stopping, evaluate on `ewt_dev` / `conll_test` / `astro_test`, write the **same** `summary.csv` schema as the iterative runs into `runs/baselines/<name>/summary.csv` so plots can overlay
3. `astro_only` uses `max_seq_len=512`, `batch_size=4`, `unit=paragraph` (paragraphs are long); the others use `max_seq_len=256`, `batch_size=16`, `unit=sentence`

**State Management:**
- No mutable global state. Config is passed as `argparse.Namespace` (baseline) or frozen `ExperimentConfig` (iterative)
- Best-checkpoint state lives only in `<output_dir>/best_model/` on disk and is reloaded explicitly
- Random state seeded once per seed in the iterative driver (`scripts/run_experiment.py:195`)

## Key Abstractions

**`Example` dict** (the canonical in-memory unit):
- Purpose: Pre-tokenisation representation of one sentence or paragraph
- Examples: every list returned by `data.parse_iob2`
- Pattern: `{ "id": "<file_stem>_<idx:05d>", "tokens": [...], "ner_tags": [...], "raw_lines": [...], "unit": "sentence" | "paragraph" }`
- Defined in: `data.py:71-79`
- Used by: tokenisation, injection pool, prediction file writing

**`ExperimentConfig`** (frozen dataclass tree):
- Purpose: Immutable container for one iterative experiment
- Examples: result of `load_config("experiments/config_conll.yaml")`
- Pattern: nested `SourceCfg`, `TargetCfg`, `List[EvalSetCfg]` — all `@dataclass(frozen=True)` so downstream code can never mutate config mid-run
- Defined in: `src/experiment/config_loader.py:18-66`

**Unified label set:**
- Purpose: One label space across three datasets so a single classification head fits all
- Pattern: `LABEL_LIST` (65 entries) + `UNIFY_MAP` (WIESP Person/Organization/Location → PER/ORG/LOC) + `normalize_tag` (drops MISC, falls back to `O` for unknowns)
- Defined in: `config.py:4-42`, `data.py:12-35`

**Run-dir layout** (output convention):
- Purpose: Deterministic, hierarchical artefact storage so plotting code can scan `runs/`
- Pattern: `runs/<experiment_name>/seeds/seed_<S>/iter_<NNN>/{meta,metrics,per_type_metrics}.json` + per-eval-set artefacts; one `summary.csv` per experiment
- Defined in: `src/experiment/logging_io.py:110-125`

**`SUMMARY_COLUMNS` schema:**
- Purpose: Stable, plot-friendly CSV row schema covering both the iterative experiment and the per-domain baselines
- Pattern: List of explicit column names, including `f1_<TYPE>` and `support_<TYPE>` for every BIO-collapsed entity type
- Defined in: `src/experiment/logging_io.py:39-51`

## Entry Points

**`baseline_main_v1.py`:**
- Location: `baseline_main_v1.py`
- Triggers: `python baseline_main_v1.py [--data_dir ... --epochs ... --final_eval]`
- Responsibilities: Train one BERT on EWT, evaluate cross-domain on dev (and optionally test), save EWT test predictions for LearnIT submission

**`scripts/run_experiment.py`:**
- Location: `scripts/run_experiment.py`
- Triggers: `python scripts/run_experiment.py --config experiments/config_{conll,astro}.yaml [--device cuda] [--debug]`
- Responsibilities: Iterative cross-domain protocol (one fresh model per `(seed, iter)` cell over a target-injection schedule); writes per-iter artefacts and a shared `summary.csv`

**`scripts/run_baselines.py`:**
- Location: `scripts/run_baselines.py`
- Triggers: `python scripts/run_baselines.py [--baselines ...] [--seeds ...]`
- Responsibilities: Train-once per-domain baselines with the *same* output schema for overlay plotting

**HPC entry points (SLURM):**
- `hpc/install_env.job` — one-time conda env creation (`scavenge` partition, no GPU)
- `hpc/smoke_test.job` — 30-min `--debug` sanity run on any GPU
- `hpc/train_iter.job` — real training; pinned to `acltr` partition + V100; takes `CFG=...` env var to choose YAML

## Architectural Constraints

- **Threading:** Single-process, single-GPU. PyTorch `DataLoader` defaults to single-worker (no `num_workers` set anywhere). Tokenisation runs in-process via `datasets.Dataset.map(batched=True)` (`data.py:236-242`).
- **Global state:** None mutable. Module-level constants only: `LABEL_LIST`, `LABEL2ID`, `ID2LABEL` in `config.py`; `UNIFY_MAP` in `data.py:12-20`; `_IOB2_COLS` and `BASELINES` dicts in the script files; `ENTITY_TYPES` and `SUMMARY_COLUMNS` derived once at import in `src/experiment/logging_io.py`.
- **Circular imports:** None. `src/experiment/` always imports *down* into top-level (`trainer`, `config`, `data`, `model`); top-level never imports from `src/experiment/`.
- **`sys.path` injection:** Both scripts in `scripts/` prepend the repo root to `sys.path` so plain `import data` / `import trainer` works regardless of cwd (`scripts/run_experiment.py:20-22`, `scripts/run_baselines.py:26-28`). New scripts must follow this pattern.
- **Determinism:** `trainer.set_seed` seeds `random`, `numpy`, `torch`, and `torch.cuda` (`trainer.py:17-21`). cuDNN determinism flags are *not* set, so bit-exact reproducibility across runs of the same seed on the same GPU is not guaranteed; F1 should still be GPU-invariant per the HPC README.
- **HF cache:** `HF_HOME` is exported in HPC jobs to `~/.cache/huggingface` to avoid re-downloading `bert-base-cased` on every iteration.
- **`numpy<2` pin:** Required because `seqeval` and some older `transformers` paths still misbehave on numpy 2 (see `requirements.txt` and `hpc/README_HPC.md` "Common gotchas").

## Anti-Patterns

### Mutating model across iterations

**What happens:** A naive iterative loop would keep a single `model` object across iterations, fine-tuning it incrementally on growing target mixes.
**Why it's wrong:** The experimental protocol is "fresh fine-tune from pretrained per cell". Re-using a model conflates iteration-level adaptation with cumulative training time and breaks the comparison.
**Do this instead:** `model = build_model(cfg.model_name).to(device)` inside the iteration loop on every iter (`scripts/run_experiment.py:224`). The comment `# FRESH MODEL — DO NOT MOVE OUTSIDE LOOP` is load-bearing.

### Reading/writing run artefacts non-atomically

**What happens:** A long training run can crash mid-write, leaving truncated `summary.csv` or `metrics.json`. Tools downstream then read corrupt rows.
**Why it's wrong:** Crashes in the middle of fine-tuning are common (HPC time-outs, OOM). Partial writes silently corrupt the run.
**Do this instead:** Use `src/experiment/logging_io.py:write_json` / `write_jsonl` / `write_text` / `write_confusion_csv` / `append_summary_row` — they all go through `tmp + os.replace` (`src/experiment/logging_io.py:56-74`).

### Bypassing `normalize_tag` when reading IOB2

**What happens:** Adding a new dataset and feeding its raw NER tags directly into `LABEL2ID` lookup raises `KeyError` for any unmapped tag.
**Why it's wrong:** The unified label set (`config.py:LABEL_LIST`) is closed; per-dataset variants need to be remapped.
**Do this instead:** Always go through `data.normalize_tag(tag)` before any `LABEL2ID[...]` lookup. Extend `data.py:UNIFY_MAP` for new overlapping concepts, and extend `LABEL_LIST` for genuinely new types.

### Two `parse_iob2` implementations

**What happens:** `src/helper/helper_funcs.py:parse_iob2` is an *older, simpler* parser used only by `eda.ipynb` (no `id`, no `raw_lines`, no `unit`). New training/eval code must not import it.
**Why it's wrong:** It returns a different example schema and doesn't support paragraph mode, which silently breaks the iterative pipeline.
**Do this instead:** Always import `parse_iob2` from `data.py` (`from data import parse_iob2`). Treat `src/helper/helper_funcs.py` as EDA-only.

### Re-tokenising eval splits inside the iteration loop

**What happens:** Re-running `prepare_split` for the eval sets every iteration wastes minutes of HPC time per cell.
**Why it's wrong:** The eval splits and target dev never change across iterations of one experiment.
**Do this instead:** Tokenise eval + dev once before the seed loop and reuse the loaders (`scripts/run_experiment.py:160-174`).

## Error Handling

**Strategy:** Fail fast on configuration / data errors, tolerate environment edge cases.

**Patterns:**
- CLI / config validation: `argparse` does the work; `data.parse_iob2` raises `ValueError` for invalid `unit` (`data.py:64-65`)
- Device fallback: if `--device cuda` and `torch.cuda.is_available()` is False, fall back to CPU with a print (`baseline_main_v1.py:34-37`, `scripts/run_experiment.py:122-123`, `scripts/run_baselines.py:286-287`)
- File IO: atomic writes prevent corruption; `git_commit_hash` swallows exceptions and returns `"unknown"` (`src/experiment/config_loader.py:121-129`)
- Training: no try/except inside the train loop — any error (OOM, NaN, bad data) propagates and aborts the run, which is the right behaviour for an experiment
- Early stopping: `trainer.train` honours `early_stopping_patience` (`trainer.py:240-245`); `None` disables it

## Cross-Cutting Concerns

**Logging:** `print` to stdout (with section banners using box-drawing characters) plus structured JSONL train logs (`trainer.py:223-232`). HPC jobs export `PYTHONUNBUFFERED=1` so logs flush in real time (`hpc/train_iter.job:56`). No `logging` module use.

**Validation:** Implicit only — IOB2 parser silently skips malformed lines (`data.py:90-94`); tokenisation uses `truncation=True` and counts truncated examples via the optional `trunc_counter` (`data.py:211-214`); paragraph truncation >10% triggers a `WARNING` print (`scripts/run_experiment.py:90-92`).

**Authentication:** None — local files only. HF model downloads use the public `google-bert/bert-base-cased` checkpoint.

**Reproducibility:**
- Seeded everywhere: `random`, `numpy`, `torch`, `torch.cuda` (`trainer.py:17-21`)
- Per-run snapshot: dataset SHA256s + git commit + machine info + tokenizer class (`src/experiment/logging_io.py:130-150`)
- Injection order persisted to disk before any training starts (`src/experiment/injection.py:31-37`)
- `runs/**/predictions_*.jsonl` is gitignored (large, regeneratable from the same seed); everything else (`summary.csv`, `metrics.json`, `train_log.jsonl`, etc.) is committed.

---

*Architecture analysis: 2026-05-07*

# Coding Conventions

**Analysis Date:** 2026-05-07

## Naming Patterns

**Files:**
- Top-level entry points/modules use flat lowercase names with optional underscores: `baseline_main_v1.py`, `config.py`, `data.py`, `model.py`, `trainer.py`
- Versioned entry points embed the version in the filename: `baseline_main_v1.py`
- Reusable library code lives in `src/<package>/` with snake_case modules: `src/experiment/config_loader.py`, `src/experiment/logging_io.py`, `src/experiment/evaluation.py`, `src/experiment/injection.py`, `src/helper/helper_funcs.py`
- Driver scripts under `scripts/` are prefixed with `run_`: `scripts/run_experiment.py`, `scripts/run_baselines.py`
- Experiment configs are YAML, prefixed with `config_`: `experiments/config_conll.yaml`, `experiments/config_astro.yaml`
- HPC SLURM jobs use `.job` extension under `hpc/`: `hpc/train_iter.job`, `hpc/install_env.job`, `hpc/smoke_test.job`

**Functions:**
- Public functions: `snake_case` — `parse_iob2`, `load_all_datasets`, `prepare_split`, `make_dataloader`, `save_predictions`, `build_model`, `evaluate`, `train`, `set_seed`, `build_injection_pool`, `full_evaluate`, `load_config`
- Private/internal helpers: `_leading_underscore` — `_decode_batch`, `_decode_batch_full` (`trainer.py`), `_flush` (nested in `data.py:71`), `_atomic_write` (`src/experiment/logging_io.py:56`), `_bio_collapse` (`src/experiment/evaluation.py:17`), and the `_load`, `_build_eval_loader`, `_cols_for`, `_load_examples`, `_print_truncation`, `_trainer_args`, `_eval_dataset_name`, `_eval_unit` cluster in `scripts/`
- Closures returned by factory functions follow the same `snake_case` style (e.g. `tokenize_and_align_labels` returned from `make_tokenize_fn` in `data.py:184`)

**Variables:**
- Locals: `snake_case` (`train_loader`, `dev_loader`, `best_dev_f1`, `epochs_no_improve`, `target_chunk_ids`)
- Aliased imports preserve readability at call sites: `from trainer import set_seed as seed_everything` (`baseline_main_v1.py:8`, `scripts/run_experiment.py:34`, `scripts/run_baselines.py:40`); `from trainer import train as trainer_train`
- Module-level configuration constants: `UPPER_SNAKE_CASE` — `LABEL_LIST`, `LABEL2ID`, `ID2LABEL` (`config.py`); `UNIFY_MAP` (`data.py:12`); `BASELINES`, `EVAL_SETS`, `DEFAULTS` (`scripts/run_baselines.py`); `_IOB2_COLS`, `SUMMARY_COLUMNS`, `ENTITY_TYPES` (`src/experiment/`)
- Throwaway iteration variables use single letters when the meaning is local (`t`, `l`, `p`, `g`) — e.g. `src/experiment/evaluation.py:50-51`

**Types / dataclasses:**
- `PascalCase`: `SourceCfg`, `TargetCfg`, `EvalSetCfg`, `ExperimentConfig` in `src/experiment/config_loader.py`
- All config dataclasses are `@dataclass(frozen=True)` so configuration cannot be mutated mid-run

## Code Style

**Formatting:**
- No formatter is configured. There is no `pyproject.toml`, `.pre-commit-config.yaml`, `ruff`, `black`, or `isort` config in the repo.
- Manual style is consistent but informal: 4-space indentation, double-quoted strings, vertical alignment of `=` and call kwargs is used liberally for readability (see the dataset-loading dict in `data.py:146-162` and the dataloader assignments in `baseline_main_v1.py:57-63`). Alignment takes priority over a strict line-length cap.

**Linting:**
- No linter configuration committed (`.flake8`, `ruff.toml`, `pylintrc`, `mypy.ini` all absent).
- `.gitignore` reserves slots for `.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/` but none of these tools are wired up.

## Import Organization

**Order (consistent across the codebase):**
1. Python standard library (`os`, `json`, `time`, `random`, `argparse`, `csv`, `hashlib`, `subprocess`, `platform`, `shutil`, `sys`, `pathlib`, `types`, `typing`, `dataclasses`)
2. Third-party packages (`numpy`, `torch`, `transformers`, `tqdm`, `seqeval`, `datasets`, `sklearn`, `yaml`)
3. First-party project modules (`config`, `data`, `model`, `trainer`, `src.experiment.*`)

Examples:
- `trainer.py:1-14` — stdlib → 3rd party → blank line → `from config import ID2LABEL`
- `scripts/run_experiment.py:11-59` — stdlib → `sys.path` repo-root injection → torch/transformers → `from data import …` → `from src.experiment.* import …`

**Path manipulation for scripts:**
Scripts under `scripts/` insert the repo root onto `sys.path` so that flat imports work regardless of cwd:
```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```
Verbatim in `scripts/run_experiment.py:20-22` and `scripts/run_baselines.py:26-28`.

**Path Aliases:**
- No `setup.py` and no `pyproject.toml`. The project is not installable; everything is run from the repo root with the `sys.path` shim above.
- `src/experiment/__init__.py` exists but is empty (1 byte). `src.experiment` is a namespace-style package with explicit submodule imports.

## Type Hints

- Type hints are present but inconsistent. Most public APIs in `data.py`, `src/experiment/config_loader.py`, `src/experiment/evaluation.py`, `src/experiment/injection.py`, and `src/experiment/logging_io.py` are annotated. `model.py:build_model`, `trainer.py:_decode_batch_full`, and several closures in `data.py` are unhinted or partially hinted.
- `from __future__ import annotations` is **not** used.
- Containers use the typing-module forms (`List`, `Dict`, `Tuple`, `Optional`), always imported as `from typing import List, Dict, Tuple, Optional`.

## Error Handling

**Patterns:**
- The codebase uses very few exceptions. Only two `try/except` blocks exist, both in `src/experiment/config_loader.py` (lines 122-129 around `git rev-parse HEAD`, lines 137-144 around the optional `import torch`). Both swallow all exceptions and degrade to a sentinel value (`"unknown"`) so snapshot capture cannot crash a training run.
- The only explicit `raise` is an argument validation in `data.py:65`:
  ```python
  if unit not in ("sentence", "paragraph"):
      raise ValueError(f"unit must be 'sentence' or 'paragraph', got {unit!r}")
  ```
- Soft fallbacks for environment problems print and continue rather than raise:
  ```python
  if args.device == "cuda" and not torch.cuda.is_available():
      print("CUDA not available, falling back to CPU.")
      args.device = "cpu"
  ```
  See `baseline_main_v1.py:34-37`, `scripts/run_experiment.py:121-123`, `scripts/run_baselines.py:286-287`.
- Truncation warnings are non-fatal and printed inline (`scripts/run_baselines.py:151-152`, `scripts/run_experiment.py:90-92`):
  ```python
  if e["unit"] == "paragraph" and pct > 10.0:
      print(f"  !! WARNING: >10% astro paragraphs truncated for {e['name']}")
  ```

**Rule of thumb:** Validate at the boundary (config loader, CLI, file format), let everything inside crash loudly if invariants break.

## Logging

**Framework:**
- Python's `logging` module is **not used**. All progress/status output goes through `print()` (51 `print(` call sites).
- Training-loop progress bars use `tqdm` (`trainer.py:84, 192`).

**Patterns:**
- Section headers use a Unicode box-drawing run:
  ```python
  print("\n── Loading datasets ─────────────────────────────────────")
  ```
  Used consistently in `baseline_main_v1.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`.
- Phase separators use double-rule equals or thick rules:
  ```python
  print(f"\n══════════ seed {seed} ══════════")           # scripts/run_experiment.py:194
  print("\n" + "=" * 60)                                  # baseline_main_v1.py:14
  ```
- Per-step diagnostic prints use 2-space indentation under the section header.
- A debug banner is printed when `--debug` is on (`scripts/run_experiment.py:117`, `scripts/run_baselines.py:282`).
- **Persistent training logs** are written as JSONL inside `trainer.train` (`trainer.py:223-232`): one line per epoch with `{epoch, train_loss, dev_loss, dev_f1, lr, grad_norm}`.
- All structured experiment artifacts (`summary.csv`, `metrics.json`, `meta.json`, `config.json`, confusion matrices, predictions JSONL) are written through `src/experiment/logging_io.py`, never through `print`.

## Configuration

**Two-tier configuration:**

1. **Argparse for the legacy single-shot baseline** (`config.py:get_args` → `baseline_main_v1.py`). Returns an `argparse.Namespace`. Defaults are inline (`epochs=5`, `batch_size=16`, `lr=5e-5`, `seed=42`). Boolean flags use `action="store_true"`.
2. **YAML + frozen dataclasses for the iterative experiments** (`src/experiment/config_loader.py`). YAML files in `experiments/` (e.g. `experiments/config_conll.yaml`) are parsed by `load_config()` into a `frozen=True` `ExperimentConfig`. The script-level argparse is reduced to `--config`, `--device`, `--debug` (`scripts/run_experiment.py:106-110`).

**Argument shape inside `trainer.train`:**
The trainer accepts a duck-typed `args` object with attributes `lr`, `weight_decay`, `warmup_ratio`, `epochs`, `output_dir`. Both code paths construct this differently:
- Argparse path: passes `args` directly from `parser.parse_args()`
- YAML path: builds a `types.SimpleNamespace` via `_trainer_args(cfg, output_dir)` (`scripts/run_experiment.py:95-102`, `scripts/run_baselines.py:188-194`)

**Label set:**
The single source of truth lives in `config.py:4-39` (`LABEL_LIST` plus derived `LABEL2ID`, `ID2LABEL`). Every other module imports from there.

## Reproducibility

**Seeding:**
- Centralised in `trainer.set_seed(seed)` (`trainer.py:17-21`):
  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  ```
- Imported under the alias `seed_everything` to make call sites obvious.
- Called once per seed in `scripts/run_experiment.py:195` and `scripts/run_baselines.py:180`, **before** building the model and dataloaders for that seed.
- HuggingFace's `transformers.set_seed` is imported in `baseline_main_v1.py:3` but the project's own `set_seed` is what actually seeds.
- **Not set:** `torch.backends.cudnn.deterministic`, `torch.backends.cudnn.benchmark`, `torch.use_deterministic_algorithms`, `PYTHONHASHSEED`. The HPC job exports `PYTHONUNBUFFERED=1` and `TOKENIZERS_PARALLELISM=false` (`hpc/train_iter.job:56-57`) but no determinism env vars.
- **Independent RNGs for ordering:** `src/experiment/injection.py:26` uses a local `random.Random(seed)` to shuffle the target injection pool, decoupling pool ordering from any other consumer of the global RNG.

**Reproducibility audit trail (per run):**
- `runs/<exp>/config.json` — full config + dataset SHA256 hashes + git commit + tokenizer info + machine info, written via `write_config_snapshot` (`src/experiment/logging_io.py:130-150`).
- `runs/<exp>/seeds/seed_<n>/injection_order.json` — `{seed, n, order_hash, ids}` for the deterministic injection pool (`src/experiment/injection.py:34-38`).
- `runs/<exp>/seeds/seed_<n>/iter_<k>/added_target_ids.txt` — exactly which target ids were added at iteration `k`.
- `runs/<exp>/seeds/seed_<n>/iter_<k>/train_log.jsonl` — per-epoch training metrics.

## File I/O

**Atomic writes:**
Every JSON/JSONL/CSV/text writer in `src/experiment/logging_io.py` follows the tmp-then-`os.replace` pattern:
```python
def _atomic_write(path: str, data: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
    os.replace(tmp, path)
```
Applies to `write_json`, `write_jsonl`, `write_text`, `append_summary_row`, `write_confusion_csv`, and the injection-order writer in `src/experiment/injection.py:34-37`. A crash mid-write therefore never corrupts a previously good file.

**`summary.csv` is append-only-by-rewrite:**
`append_summary_row` (`src/experiment/logging_io.py:83-105`) reads existing rows, adds the new one, writes everything to `summary.csv.tmp`, then `os.replace`s. This guarantees a consistent header even when columns evolve.

**Encoding:**
- All text I/O explicitly passes `encoding="utf-8"` (read and write).
- File-handle variable names: `fh` (preferred, used in `logging_io.py`, `config_loader.py`, `injection.py`) or `f` (used in `data.py`, `trainer.py`).

## Comments

**When to Comment:**
- Module-level docstrings explain the file's purpose in 2-6 lines (`model.py:1-6`, `src/experiment/config_loader.py:1-6`, `src/experiment/evaluation.py:1-7`, `scripts/run_experiment.py:1-9`, `scripts/run_baselines.py:1-15`, `src/experiment/injection.py:1-6`, `src/experiment/logging_io.py:1-7`).
- Public functions get docstrings with informal `Args:` / `Returns:` blocks (no Sphinx/numpy/google directives), e.g. `data.py:42-63`, `data.py:120-137`, `trainer.py:139-155`.
- Inline `# comments` flag (1) data-format quirks (`# EWT uses token_col=1, tag_col=2 (5-column format)` in `data.py:122`); (2) safety-critical loop placement (`# FRESH MODEL — DO NOT MOVE OUTSIDE LOOP. Re-initialised from the pretrained checkpoint at every iteration.` in `scripts/run_experiment.py:222-223`); (3) section dividers (`# ── Train ───…`, `# --- atomic JSON / text writers ---`).
- Commented-out code is occasionally left for context, e.g. the original `jaccard_vocab` body in `src/helper/helper_funcs.py:20-23`.

**Docstring style:**
Free-form prose with optional `Args:` / `Returns:` blocks. No type repetition. Returns dicts are documented inline by key shape:
```
Returns:
    dict with keys:
      best_model_dir, best_epoch, best_dev_f1,
      train_time_sec, peak_gpu_mem_mb, epochs_run
```
(`trainer.py:151-156`)

## Function Design

**Size:**
- Most public functions are 10-50 lines. The two outliers are intentional orchestrators: `trainer.train` (~125 lines, `trainer.py:134-258`) and `scripts/run_experiment.py:main` (~215 lines). Both are end-to-end pipelines that would lose clarity if split further.

**Parameters:**
- Public training/eval entry points (`evaluate`, `train`, `full_evaluate`, `run_one_baseline`) accept positional `model, dataloader, device` then keyword-only options.
- Optional flags use defaults that preserve existing behaviour. Example: `prepare_split(..., return_truncation_count=False)` in `data.py:222-245` returns a `Dataset` by default and `(Dataset, n_truncated)` when the flag is set.
- Mutable accumulators are passed in as `Optional[List[int]]` "single-element list" sentinels (the `trunc_counter` pattern in `data.py:173, 234`).

**Return Values:**
- Functions that produce metrics/results return **dicts with stable key names**, never tuples. See `evaluate` (`trainer.py:98-132`), `train` (`trainer.py:251-258`), `full_evaluate` (`src/experiment/evaluation.py:75-89`).
- Functions that build training rows return dicts keyed by the CSV schema (`build_summary_row`, `src/experiment/logging_io.py:169-205`).

## Module Design

- No `__all__` declarations.
- `src/experiment/__init__.py` is empty (1 byte). Submodules are imported by full path: `from src.experiment.evaluation import full_evaluate`.

**Cross-module dependencies:**
- `config.py` is depended on by everything (label set + argparse). It depends on nothing project-internal.
- `data.py`, `model.py`, `trainer.py` are flat top-level modules, each importing from `config`.
- `src/experiment/*` modules import from the flat top-level modules (`from data import …`, `from trainer import evaluate as trainer_evaluate`) but not vice versa — the iterative experiment layer is built on top of the legacy flat layer.
- `src/helper/helper_funcs.py` is a legacy duplicate of `data.parse_iob2` (no `id`, `raw_lines`, `unit` fields, simpler signature). Used by exploratory notebook code only — not imported by any committed `.py` script.

No barrel files / re-exports.

---

*Convention analysis: 2026-05-07*

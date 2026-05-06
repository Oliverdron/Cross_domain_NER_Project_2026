# Coding Conventions

## Summary

This is a research ML codebase for cross-domain NER with BERT. Style is clean but informal — no linter or formatter config is present. The dominant patterns are: module-level docstrings on all non-trivial files, Google-style Args/Returns docstrings on public functions, snake_case throughout, and `argparse` for CLI configuration. Private/internal helpers are prefixed with a single underscore.

## Naming Patterns

**Files:**
- `snake_case.py` for all modules (e.g., `data.py`, `trainer.py`, `config_loader.py`, `logging_io.py`)
- Script entrypoints in `scripts/` named as `run_<purpose>.py`
- Experiment YAML configs named `config_<target>.yaml`

**Functions:**
- Public functions: `snake_case` (e.g., `parse_iob2`, `build_model`, `prepare_split`, `full_evaluate`)
- Private/internal helpers: `_snake_case` with leading underscore (e.g., `_decode_batch`, `_flush`, `_atomic_write`, `_cols_for`, `_bio_collapse`)
- Closures and inner functions also prefixed with underscore (e.g., `_flush` inside `parse_iob2`)

**Variables:**
- `snake_case` universally
- Short loop variables are acceptable (`p`, `l`, `k`, `t`, `fh`, `ex`)
- Accumulator patterns use `_sum` / `_count` suffixes (e.g., `grad_norm_sum`, `grad_norm_count`)
- Counter lists used as mutable closures: `trunc_counter = [0]`

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants (e.g., `LABEL_LIST`, `LABEL2ID`, `ID2LABEL`, `UNIFY_MAP`, `SUMMARY_COLUMNS`, `ENTITY_TYPES`, `BASELINES`)

**Classes:**
- `PascalCase` for dataclasses (e.g., `SourceCfg`, `TargetCfg`, `ExperimentConfig`, `EvalSetCfg`)

**Type hints:**
- Used consistently on all public function signatures in `data.py`, `trainer.py`, `src/experiment/`
- Return types annotated (e.g., `-> List[Dict]`, `-> dict`, `-> str`)
- `Optional[...]` used for nullable params; `typing` module imported explicitly

## Code Style

**Formatting:**
- No `.prettierrc`, `.editorconfig`, or `pyproject.toml` with formatter config detected
- Indentation: 4 spaces throughout
- Line length: not enforced, but most lines stay under ~100 characters
- Alignment padding: used liberally for dict/kwarg readability (e.g., `"B-PER", "I-PER",` columns aligned in `config.py`, `LABEL2ID = {l: i ...}` aligned)

**Linting:**
- No `.flake8`, `.pylintrc`, or `ruff.toml` detected — no enforced linting

**Visual separators:**
- Section headers use `# ── Section name ──────────────────────` style (Unicode em-dash + hyphens)
- Used in `data.py`, `trainer.py`, `baseline_main_v1.py`, and `scripts/` to divide logical sections

## Import Organization

**Order (consistent across files):**
1. Standard library (`os`, `json`, `time`, `random`, `hashlib`, `csv`, `platform`, `subprocess`)
2. Third-party packages (`torch`, `numpy`, `tqdm`, `transformers`, `datasets`, `seqeval`, `sklearn`, `yaml`)
3. Local project imports (`from config import ...`, `from data import ...`, `from trainer import ...`, `from src.experiment...`)

**Path management for scripts:**
- `scripts/` files manually insert `ROOT` (resolved parent of `__file__`) into `sys.path` so flat imports like `import data` work regardless of cwd

**No `__init__.py` barrel files** except `src/experiment/__init__.py` (empty)

## Docstring Style

**Module-level docstrings:** Present on all non-trivial files. One-line description, then a paragraph of context. Example from `model.py`:
```python
"""
model.py
--------
Builds the BERT token classification model.
Kept minimal — all training logic lives in trainer.py.
"""
```

**Function docstrings:** Google-style with `Args:` and `Returns:` sections for public functions with non-obvious signatures. Example from `data.py`:
```python
"""
Parse a .iob2 file into a list of example dicts.

Args:
    filepath:  path to the .iob2 file
    token_col: column index for the token
    ...

Returns list of dicts:
    {"id": ..., "tokens": [...], ...}
"""
```

**Short private helpers:** Single-line or two-line docstrings, no Args/Returns sections (e.g., `_decode_batch`, `_flush`, `_bio_collapse`)

**Inline comments:** Used for non-obvious logic; often short (`# first subword of word`, `# NaN-equivalent in CSV`, `# truncate any pre-existing log`)

## Error Handling

**Validation:** `raise ValueError(f"... got {value!r}")` pattern used for invalid enum-like arguments (e.g., `unit` parameter in `parse_iob2`)

**Graceful fallbacks:** CUDA availability checked and device falls back to CPU with `print()` warning (no exception raised)

**File I/O:** All writes go through tmp-file + `os.replace()` atomic pattern to prevent partial writes on crash (see `src/experiment/logging_io.py`)

**Exception swallowing:** `except Exception: return "unknown"` used in `git_commit_hash()` and `machine_info()` — acceptable for non-critical metadata collection

## Logging

**Framework:** `print()` to stdout only — no `logging` module used anywhere

**Patterns:**
- Section headers: `print("\n── Section name ──────────────")`
- Progress: `tqdm` wraps all DataLoader loops with descriptive `desc=` strings
- Per-batch: `progress.set_postfix(loss=f"{loss.item():.4f}")`
- Inline prints for key metrics every epoch (`Dev F1`, `Dev loss`, `→ Best model saved`)
- Warnings prefixed with `!! WARNING:` in print output

## Configuration Pattern

**CLI config:** `argparse` in `config.py` via `get_args()` — used by `baseline_main_v1.py`

**YAML config:** `src/experiment/config_loader.py` loads `experiments/config_*.yaml` into frozen `@dataclass` objects. Config is immutable after load. Schema is explicit (no `**kwargs` passthrough).

**`SimpleNamespace`:** Used in `scripts/` to pass training hyperparams to `trainer.train()` when not using `argparse` args directly (bridges YAML config to trainer interface)

## Function Design

**Single responsibility:** Functions are small and focused. `trainer.py` cleanly separates `_decode_batch`, `evaluate`, and `train`. `data.py` separates parsing, tokenization, and DataLoader creation.

**Return dicts for multi-value returns:** Functions returning multiple metrics return a `dict` with named keys rather than tuples (e.g., `evaluate()` returns `{"f1": ..., "loss": ..., "predictions": ...}`)

**Mutable accumulator trick:** `trunc_counter = [0]` (single-element list) used as closure-compatible counter — avoids `nonlocal` in nested functions

## Module Design

**Flat imports preferred:** Top-level modules (`config`, `data`, `model`, `trainer`) are imported directly, not through a package namespace

**`src/experiment/` package:** Contains more advanced experiment utilities with clear module docstrings and separation of concerns (config loading, injection pool, evaluation, logging I/O)

**`src/helper/helper_funcs.py`:** Legacy/prototype code — older, simpler `parse_iob2` without docstrings, with a commented-out function. Style is notably less polished than main modules.

---

*Convention analysis: 2026-05-05*

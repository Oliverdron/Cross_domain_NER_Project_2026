# Codebase Structure

**Analysis Date:** 2026-05-07

## Directory Layout

```
Cross_domain_NER_Project_2026/
├── baseline_main_v1.py           # Single-train baseline entry point
├── config.py                     # CLI args + unified 65-label set
├── data.py                       # IOB2 parse + tokenise + DataLoader + save preds
├── model.py                      # build_model() — BERT token-classification head
├── trainer.py                    # train() / evaluate() / set_seed()
├── eda.ipynb                     # Exploratory data analysis notebook
├── BASELINE_V1.md                # Baseline-run instructions (project README-equiv.)
├── requirements.txt              # pip dependencies (numpy<2 pinned)
├── .gitignore                    # ignores runs/**/predictions_*.jsonl, hpc/logs/
│
├── data/                         # Local IOB2 datasets (committed)
│   ├── universal_train.iob2      # EWT (5-col format, token_col=1, tag_col=2)
│   ├── universal_dev.iob2
│   ├── universal_test_masked.iob2
│   ├── news_train.iob2           # CoNLL-2003 (2-col format)
│   ├── news_dev.iob2
│   ├── news_test.iob2
│   ├── astro_train.iob2          # WIESP-2022 (2-col, paragraph-blocked)
│   ├── astro_dev.iob2
│   └── astro_test.iob2
│
├── experiments/                  # YAML configs for iterative runs
│   ├── config_conll.yaml         # target = CoNLL (sentences)
│   └── config_astro.yaml         # target = Astro (paragraphs, max_seq_len=512)
│
├── scripts/                      # CLI runners (sys.path-injecting)
│   ├── run_experiment.py         # Iterative source+target injection runner
│   └── run_baselines.py          # ewt_only / conll_only / astro_only baselines
│
├── src/                          # Shared experiment toolkit
│   ├── experiment/
│   │   ├── __init__.py           # Empty marker
│   │   ├── config_loader.py      # YAML → frozen dataclass + dataset hashes
│   │   ├── injection.py          # Seed-deterministic target injection pool
│   │   ├── evaluation.py         # full_evaluate() — metrics + confusion + JSONL
│   │   └── logging_io.py         # Atomic CSV/JSON/JSONL writers, summary schema
│   └── helper/
│       └── helper_funcs.py       # Legacy parse_iob2 + jaccard_vocab (EDA only)
│
├── hpc/                          # ITU HPC SLURM job scripts
│   ├── README_HPC.md             # Step-by-step HPC usage guide
│   ├── install_env.job           # One-time: conda env `ner` (scavenge, no GPU)
│   ├── smoke_test.job            # 30-min --debug sanity check (any GPU)
│   └── train_iter.job            # Real training (acltr partition, V100)
│
└── runs/                         # Output artefacts (mostly gitignored content)
    └── <experiment_name>/
        ├── config.json           # Config snapshot + dataset SHA256 + git commit
        ├── summary.csv           # One row per (seed, iter, eval_set)
        └── seeds/
            └── seed_<S>/
                ├── injection_order.json
                └── iter_<NNN>/
                    ├── meta.json
                    ├── metrics.json
                    ├── per_type_metrics.json
                    ├── train_log.jsonl
                    ├── added_target_ids.txt
                    ├── confusion_matrix_<eval>.csv
                    └── predictions_<eval>.jsonl   (gitignored)
```

## Directory Purposes

**Repo root (top-level `.py` files):**
- Purpose: Core, reusable pipeline modules (data → model → trainer) + the simple baseline entry point
- Contains: `baseline_main_v1.py`, `config.py`, `data.py`, `model.py`, `trainer.py`
- Key files:
  - `baseline_main_v1.py` — runnable as `python baseline_main_v1.py`
  - `data.py` — every IOB2 / tokenisation / DataLoader function
  - `trainer.py` — `train()` and `evaluate()` are imported by all entry points

**`data/`:**
- Purpose: Local IOB2 datasets (committed to repo, ~25 MB total)
- Contains: 9 `.iob2` files (3 splits × 3 datasets)
- Key files:
  - `universal_*.iob2` — EWT (English Web Treebank), token_col=1, tag_col=2
  - `news_*.iob2` — CoNLL-2003, token_col=0, tag_col=1
  - `astro_*.iob2` — WIESP-2022, token_col=0, tag_col=1, paragraph-blocked
- Column-rule mapping lives in `scripts/run_experiment.py:_IOB2_COLS` and `scripts/run_baselines.py:BASELINES`

**`experiments/`:**
- Purpose: YAML configuration files for the iterative experiment driver
- Contains: One YAML per target domain
- Key files:
  - `config_conll.yaml` — target=conll, unit=sentence, max_seq_len=512, schedule has 20 cells (0 → 14000)
  - `config_astro.yaml` — target=astro, unit=paragraph, max_seq_len=512, schedule 0 → 1800
- Loaded by: `src/experiment/config_loader.py:load_config`

**`scripts/`:**
- Purpose: CLI runners that wire together the core pipeline + experiment helpers
- Contains: `run_experiment.py`, `run_baselines.py`
- Naming convention: `run_<thing>.py`
- Both files prepend the repo root to `sys.path` (`Path(__file__).resolve().parents[1]`) so plain top-level imports work from any cwd

**`src/experiment/`:**
- Purpose: Iterative-experiment-specific toolkit (not used by the simple baseline)
- Contains: `config_loader.py` (YAML loader), `injection.py` (target pool), `evaluation.py` (full metrics), `logging_io.py` (atomic writers + summary schema), `__init__.py` (empty)
- Imported as `from src.experiment.<mod> import …`

**`src/helper/`:**
- Purpose: Earlier helper functions kept around for the EDA notebook
- Contains: `helper_funcs.py` only (no `__init__.py` — usage is direct file import in the notebook)
- DO NOT import `parse_iob2` from here in new code; use `data.parse_iob2` instead

**`hpc/`:**
- Purpose: SLURM job scripts and HPC documentation for ITU's cluster
- Contains: 3 `.job` files + `README_HPC.md`
- `hpc/logs/` is created automatically on first job and is gitignored

**`runs/`:**
- Purpose: Output artefact tree (one subtree per experiment, then per seed, per iter)
- Contains: `runs/<experiment_name>/{config.json, summary.csv, seeds/seed_<S>/iter_<NNN>/...}`
- Currently empty in the working copy (only a `.DS_Store`); populated by experiment runs
- `runs/**/predictions_*.jsonl` is gitignored (large, regeneratable); everything else committed

**`__pycache__/` (root and `src/experiment/`):**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (gitignored)

**`.planning/codebase/`:**
- Purpose: Codebase analysis docs produced by the GSD-mapper
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md` (this file), and other focus-area outputs

## Key File Locations

**Entry Points:**
- `baseline_main_v1.py` — single-domain baseline, train+eval+save
- `scripts/run_experiment.py` — iterative source+target experiment
- `scripts/run_baselines.py` — three single-domain baselines (matching summary schema)
- `hpc/train_iter.job` — SLURM wrapper for `run_experiment.py`
- `hpc/smoke_test.job` — SLURM wrapper for `run_experiment.py --debug`
- `hpc/install_env.job` — SLURM wrapper for conda env creation

**Configuration:**
- `config.py` — CLI args, `LABEL_LIST` (65), `LABEL2ID`, `ID2LABEL`
- `experiments/config_conll.yaml` / `experiments/config_astro.yaml` — iterative experiment configs
- `requirements.txt` — pip deps (`numpy<2` is intentional)
- `.gitignore` — note `runs/**/predictions_*.jsonl` and `hpc/logs/` are excluded

**Core Logic:**
- `data.py` — `parse_iob2`, `normalize_tag`, `prepare_split`, `make_dataloader`, `save_predictions`, `entity_density`, `load_all_datasets`, `make_tokenize_fn`
- `model.py` — `build_model(model_name)` (one function, 8 lines)
- `trainer.py` — `set_seed`, `train`, `evaluate`, `_decode_batch`, `_decode_batch_full`
- `src/experiment/config_loader.py` — `load_config`, `ExperimentConfig`, `collect_dataset_hashes`, `git_commit_hash`
- `src/experiment/injection.py` — `build_injection_pool`, `slice_for_iter`, `select_examples`
- `src/experiment/evaluation.py` — `full_evaluate`, `_bio_collapse`
- `src/experiment/logging_io.py` — `write_json`, `write_jsonl`, `write_text`, `write_confusion_csv`, `append_summary_row`, `build_summary_row`, `write_config_snapshot`, `init_iter_dir`, `init_seed_dir`, `run_root`, `SUMMARY_COLUMNS`, `ENTITY_TYPES`

**Testing:**
- No automated test directory. The `--debug` flag (in both scripts) is the smoke-test mechanism: 0.1% of data, 1 seed, 5 iterations max. `hpc/smoke_test.job` invokes it on the cluster.

**Documentation:**
- `BASELINE_V1.md` — usage of `baseline_main_v1.py` (kept naming despite renaming the entry point from `main.py`)
- `hpc/README_HPC.md` — HPC walkthrough

## Naming Conventions

**Files:**
- Top-level Python modules: `lowercase.py` (`config.py`, `data.py`, `model.py`, `trainer.py`)
- Versioned entry points: `baseline_main_v<N>.py` (current: `baseline_main_v1.py`)
- Script runners: `scripts/run_<thing>.py` (`run_experiment.py`, `run_baselines.py`)
- YAML configs: `experiments/config_<target_domain>.yaml`
- SLURM jobs: `hpc/<purpose>.job`
- Docs: `UPPERCASE.md` for top-level (`BASELINE_V1.md`), `README_<scope>.md` inside subfolders (`hpc/README_HPC.md`)

**Functions:**
- `snake_case` for all Python functions
- Private helpers prefixed `_` (e.g., `_decode_batch`, `_atomic_write`, `_bio_collapse`, `_eval_dataset_name`)
- Builder/factory functions: `build_*` (`build_model`, `build_summary_row`, `build_injection_pool`)
- Side-effecting writers: `write_*` (`write_json`, `write_jsonl`, `write_confusion_csv`)
- Initialisers: `init_*` (`init_iter_dir`, `init_seed_dir`)

**Variables / constants:**
- Module-level constants: `UPPER_SNAKE_CASE` (`LABEL_LIST`, `LABEL2ID`, `ID2LABEL`, `UNIFY_MAP`, `SUMMARY_COLUMNS`, `ENTITY_TYPES`, `BASELINES`, `EVAL_SETS`, `DEFAULTS`)
- Local variables / function args: `snake_case`
- Dataset keys are short lowercase strings: `"ewt"`, `"conll"`, `"wiesp"` (also `"astro"` as a synonym in some scripts)

**Run-dir naming (output convention):**
- Seed dirs: `seeds/seed_<INT>/`
- Iteration dirs: `iter_<NNN>/` (zero-padded to 3 digits — `iter_000`, `iter_017`, …)
- Per-eval-set artefacts: `confusion_matrix_<evalset>.csv`, `predictions_<evalset>.jsonl`

**Label naming:**
- BIO tags: `B-<TYPE>` and `I-<TYPE>`
- Cross-dataset overlapping labels canonicalised to `PER` / `ORG` / `LOC` (uppercase, see `data.py:UNIFY_MAP`)
- WIESP domain-specific labels keep their CamelCase from the dataset (e.g. `B-CelestialObject`, `B-Telescope`)

## Where to Add New Code

**New entry point / runner script:**
- Primary code: `scripts/run_<purpose>.py`
- Must prepend repo root to `sys.path` (mirror `scripts/run_experiment.py:20-22`)
- Must include the `if __name__ == "__main__": main()` guard so importing the module never auto-trains

**New iterative experiment configuration:**
- YAML: `experiments/config_<target_name>.yaml`
- Schema: mirror `experiments/config_conll.yaml`; required keys are validated by `src/experiment/config_loader.py:load_config`
- HPC entry: pass via `sbatch --export=ALL,CFG=experiments/config_<name>.yaml hpc/train_iter.job` — no new job script needed

**New evaluation metric / artefact:**
- Implementation: extend `src/experiment/evaluation.py:full_evaluate` (returns a dict)
- Schema: add the metric to `src/experiment/logging_io.py:SUMMARY_COLUMNS` and `build_summary_row`
- Reading-side: any plotting code consumes `summary.csv` directly, so backward compat means appending columns at the end

**New dataset / IOB2 column layout:**
- Add column rule to `scripts/run_experiment.py:_IOB2_COLS` and `scripts/run_baselines.py:BASELINES`
- If new entity types: extend `config.py:LABEL_LIST` (this also extends `SUMMARY_COLUMNS` indirectly via `_bio_collapsed_types`)
- If overlapping with existing types: add to `data.py:UNIFY_MAP`

**New training-loop feature (e.g. AMP, scheduler change):**
- Extend `trainer.py:train` — keep its return-dict shape stable so `scripts/run_experiment.py` and `scripts/run_baselines.py` don't break
- If adding a new arg, add it to `args` (the `argparse.Namespace` from `config.py`) AND to the `_trainer_args` SimpleNamespace in `scripts/run_experiment.py:95-102`

**New helper / utility:**
- Reusable across experiments: `src/experiment/<thing>.py`
- One-off / EDA-only: do NOT add to `src/helper/`; put it in the notebook itself
- Top-level pipeline (used by both baseline + iterative): root-level `.py` like `data.py`

**New tests:**
- No formal test directory yet. Mimic the `--debug` flag pattern: every script supports `--debug` to run on 0.1% of data with one seed for ~30 min on the smoke-test partition (`hpc/smoke_test.job`).

## Special Directories

**`runs/`:**
- Purpose: Output artefacts from `scripts/run_experiment.py` and `scripts/run_baselines.py`
- Generated: Yes
- Committed: Partially — `summary.csv`, `metrics.json`, `meta.json`, `per_type_metrics.json`, `confusion_matrix_*.csv`, `train_log.jsonl`, `injection_order.json`, `added_target_ids.txt` are committed; `runs/**/predictions_*.jsonl` is gitignored (can hit ~30 MB per iteration on astro)

**`hpc/logs/`:**
- Purpose: SLURM `.out`/`.err` files (auto-created on first job submission)
- Generated: Yes
- Committed: No (gitignored)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (gitignored)

**`data/`:**
- Purpose: Source IOB2 corpora
- Generated: No (provided with the project)
- Committed: Yes — they are required for any run

---

*Structure analysis: 2026-05-07*

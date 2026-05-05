# External Integrations

## Summary

This project has two external integration points: HuggingFace Hub (model and tokenizer downloads at runtime) and the ITU HPC cluster (SLURM-based GPU compute). All training data is stored locally as `.iob2` files in `data/`. There are no third-party APIs, cloud storage services, or databases.

**Analysis Date:** 2026-05-05

## APIs & External Services

**HuggingFace Hub:**
- Service: `huggingface.co` model registry
- Purpose: download pretrained `google-bert/bert-base-cased` weights and tokenizer on first run
- SDK/Client: `transformers` library (`AutoModelForTokenClassification.from_pretrained(...)`, `AutoTokenizer.from_pretrained(...)`)
- Auth: none required (public model)
- Cache: `$HF_HOME` (defaults to `~/.cache/huggingface`); on HPC set to `/home/olgy/.cache/huggingface` in `hpc/train_iter.job` and `hpc/smoke_test.job` to persist across re-submissions
- Model size: ~440 MB (`bert-base-cased`)
- Env var: `TOKENIZERS_PARALLELISM=false` set in HPC jobs to suppress fork warnings

## Data Storage

**Databases:**
- None — no database is used

**Local Data Files:**
- All training, dev, and test data are local `.iob2` (tab-separated IOB2 format) files in `data/`
- Files present:
  - `data/universal_train.iob2` — EWT source domain train (5-column format, token_col=1, tag_col=2)
  - `data/universal_dev.iob2` — EWT source domain dev
  - `data/universal_test_masked.iob2` — EWT test (gold tags masked; used as source-forgetting probe only)
  - `data/news_train.iob2` — CoNLL-2003 news domain train (2-column format)
  - `data/news_dev.iob2` — CoNLL-2003 news domain dev
  - `data/news_test.iob2` — CoNLL-2003 news domain test
  - `data/astro_train.iob2` — WIESP-2022 astrophysics domain train (paragraph-blocked)
  - `data/astro_dev.iob2` — WIESP-2022 astrophysics domain dev
  - `data/astro_test.iob2` — WIESP-2022 astrophysics domain test
- Parsing: `data.py` → `parse_iob2()` reads files line-by-line; blank lines delimit sentence/paragraph boundaries

**Experiment Outputs:**
- Written locally to `runs/` (iterative experiments) and `outputs/` (legacy baseline)
- Per-iteration artifacts: `meta.json`, `metrics.json`, `per_type_metrics.json`, `confusion_matrix_*.csv`, `predictions_*.jsonl`, `train_log.jsonl`, `added_target_ids.txt`
- Aggregate: `summary.csv` (append-only, one row per eval set per iteration)
- All writes use atomic `tmp → os.replace` pattern (`src/experiment/logging_io.py`)

**File Storage:**
- Local filesystem only — no S3, GCS, or equivalent

**Caching:**
- HuggingFace model/tokenizer cache only (filesystem at `$HF_HOME`)
- No application-level caching layer

## Authentication & Identity

**Auth Provider:**
- None — no user auth, no login system
- HPC access via SSH (`ssh olgy@hpc.itu.dk`)
- GitHub repo access (referenced as `https://github.com/Oliverdron/Cross_domain_NER_Project_2026.git`): personal access token if private, standard HTTPS otherwise

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, Datadog, or equivalent

**Logs:**
- SLURM stdout/stderr: `hpc/logs/%x_%j.out` / `hpc/logs/%x_%j.err` (auto-created by SLURM)
- Per-iteration training log: `runs/<exp>/<seed>/iter_<k>/train_log.jsonl` — one JSON line per epoch with `{epoch, train_loss, dev_loss, dev_f1, lr, grad_norm}`
- Console output via `print()` and `tqdm` progress bars in `trainer.py` and scripts

## CI/CD & Deployment

**Hosting:**
- ITU HPC cluster (`hpc.itu.dk`)
- SLURM workload manager; jobs defined in `hpc/`

**SLURM Jobs:**

| File | Purpose | Partition | GPU | Time |
|------|---------|-----------|-----|------|
| `hpc/install_env.job` | One-time conda env setup | `scavenge` | none | 1h |
| `hpc/smoke_test.job` | Sanity check with `--debug` (0.1% data) | `scavenge` | V100 | 30min |
| `hpc/train_iter.job` | Full iterative training run | `acltr` | V100 32GB | 2 days |

**Submit commands:**
```bash
sbatch hpc/install_env.job
sbatch hpc/smoke_test.job
sbatch --job-name=iter_conll --export=ALL,CFG=experiments/config_conll.yaml hpc/train_iter.job
sbatch --job-name=iter_astro --export=ALL,CFG=experiments/config_astro.yaml hpc/train_iter.job
```

**CI Pipeline:**
- None — no automated CI (GitHub Actions, etc.)

**Results retrieval:**
```bash
scp -rp olgy@hpc.itu.dk:/home/olgy/Cross_domain_NER_Project_2026/runs ./runs_hpc
```

## Environment Configuration

**Required env vars (HPC jobs set these explicitly):**
- `HF_HOME` — HuggingFace cache directory (set to `/home/olgy/.cache/huggingface` in training jobs)
- `TMPDIR` — temp directory for pip/conda (set to `/home/olgy/tmp`)
- `PYTHONUNBUFFERED=1` — ensures stdout is flushed immediately to SLURM log files
- `TOKENIZERS_PARALLELISM=false` — suppresses HuggingFace tokenizer parallelism warnings in forked processes

**Secrets location:**
- None detected — no credentials, API keys, or `.env` files in the repo

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

## Gaps / Unknowns

- Exact installed package versions are not pinned (no `requirements-lock.txt` or `pip freeze` output committed). Actual versions in the `ner` conda env on HPC are not recorded in the repo.
- The `evaluate` package is installed in `hpc/install_env.job` (line: `pip install ... evaluate`) but is not imported anywhere in the current source — possibly a leftover or intended for future use.
- The GitHub repo URL (`Oliverdron/Cross_domain_NER_Project_2026`) suggests a different account than the local git user (`Samuel`); no remote `origin` was inspected directly.

---

*Integration audit: 2026-05-05*

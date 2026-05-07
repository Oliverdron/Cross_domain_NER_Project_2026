# External Integrations

**Analysis Date:** 2026-05-07

## APIs & External Services

**Pretrained Model Registry:**
- Hugging Face Hub - Hosts `google-bert/bert-base-cased`, the only pretrained backbone used by this project
  - SDK/Client: `transformers` (`AutoModelForTokenClassification.from_pretrained`, `AutoTokenizer.from_pretrained`)
  - Auth: None required — `bert-base-cased` is a public model
  - Referenced in:
    - `config.py` line 55 (default `--model_name` for the baseline)
    - `experiments/config_astro.yaml` lines 18-19 (`model_name`, `tokenizer_name`)
    - `experiments/config_conll.yaml` lines 15-16
    - `scripts/run_baselines.py` lines 97-98 (`DEFAULTS.model_name`, `DEFAULTS.tokenizer_name`)
  - Cache location on HPC: `HF_HOME=/home/olgy/.cache/huggingface` (`hpc/train_iter.job` line 52, `hpc/smoke_test.job` line 38) — first run downloads ~440 MB; re-runs hit the cache.

**Other APIs:**
- None. No REST clients, no `requests`/`httpx`/`aiohttp` imports. No cloud SDKs (AWS, GCP, Azure).

## Data Storage

**Databases:**
- None.

**File Storage:**
- Local filesystem only. All datasets, model checkpoints, predictions, logs, and run summaries live on disk.
  - Datasets: `data/*.iob2` (nine IOB2-formatted files, ~26 MB total)
  - Run outputs: `runs/<experiment_name>/seeds/seed_<N>/iter_<KKK>/` (created by `src/experiment/logging_io.py` `init_iter_dir`)
  - Best checkpoints: written by `transformers` `model.save_pretrained()` to `iter_dir/_best_hf/` (`trainer.py` line 237) and **deleted after evaluation** to save disk (`scripts/run_experiment.py` line 244, `scripts/run_baselines.py` line 210)
  - Atomic writes: every JSON/JSONL/CSV writer in `src/experiment/logging_io.py` uses a `tmp + os.replace` pattern (lines 56-61) so a crash never corrupts a partial file.

**Caching:**
- Hugging Face Hub cache at `~/.cache/huggingface` (or `HF_HOME` if set). No application-level caching beyond that.

## Authentication & Identity

**Auth Provider:**
- None. The project is local-only research code.
- Hugging Face Hub access uses **anonymous downloads** (`bert-base-cased` is public). No `HF_TOKEN`, no `huggingface_hub.login()` calls anywhere.
- HPC access uses SSH (`hpc/README_HPC.md` line 29: `ssh olgy@hpc.itu.dk`). No application-level user identity.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, Rollbar, etc.

**Logs:**
- SLURM stdout/stderr captured to `hpc/logs/%x_%j.out` and `hpc/logs/%x_%j.err` (`hpc/train_iter.job` lines 15-16)
- Per-iteration training logs written as JSONL to `iter_dir/train_log.jsonl` by `trainer.train()` (`trainer.py` lines 223-232) — one record per epoch with `train_loss`, `dev_loss`, `dev_f1`, `lr`, `grad_norm`
- Run-level CSV summary at `runs/<experiment_name>/summary.csv` (`src/experiment/logging_io.py` `append_summary_row`, schema in `SUMMARY_COLUMNS` lines 39-51)
- Console progress via `tqdm` (`trainer.py` lines 84, 192)

**Metrics:**
- Computed locally with `seqeval` (entity-level P/R/F1, classification report) and `scikit-learn` (token-level confusion matrix + macro F1). Not exported to any external metrics service.

## CI/CD & Deployment

**Hosting:**
- ITU HPC SLURM cluster (`hpc.itu.dk`). Project is **not deployed** as a service — it is batch-executed.

**CI Pipeline:**
- None. No GitHub Actions, no `.github/workflows/`, no Travis/CircleCI config. The `.gitignore` has standard Python entries only.

**Deployment Workflow (manual, documented in `hpc/README_HPC.md`):**
1. Push code from laptop: `git push`
2. SSH to HPC, `git clone` / `git pull`
3. Submit jobs: `sbatch hpc/install_env.job` (one-time), `sbatch hpc/smoke_test.job`, `sbatch --export=ALL,CFG=... hpc/train_iter.job`
4. `scp` results back: `scp -rp olgy@hpc.itu.dk:.../runs ./runs_hpc`

## Environment Configuration

**Required env vars:**
- None for the baseline / experiment runners themselves. The Python code reads only CLI flags (`config.py`) and YAML configs (`src/experiment/config_loader.py`).
- HPC-only (set in SLURM job scripts):
  - `HF_HOME` - Persistent Hugging Face cache directory
  - `TMPDIR` - Conda/pip scratch space (`/home/olgy/tmp`)
  - `PYTHONUNBUFFERED=1` - Stream Python output to SLURM logs
  - `TOKENIZERS_PARALLELISM=false` - Suppress HF tokenizer fork warnings
  - `CFG` - Path to YAML config, passed via `sbatch --export=ALL,CFG=...`

**Secrets location:**
- No secrets file present. No `.env`, no `credentials.*`, no API keys in the repo. Hugging Face downloads are anonymous; HPC auth is via SSH outside the codebase.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- SLURM email notifications only (`hpc/train_iter.job` lines 27-28: `--mail-type=BEGIN,END,FAIL,TIME_LIMIT_80`, `--mail-user=olgy@itu.dk`). Not application-level.

## Datasets (External Inputs)

All three NER datasets are pre-downloaded as `.iob2` files in `data/` and committed/distributed alongside the code (no runtime download). They are loaded by `data.parse_iob2()` (`data.py` lines 40-97):

- **EWT (Universal Dependencies English Web Treebank)** - Source domain (web text: blogs, forums, emails)
  - Files: `data/universal_train.iob2`, `data/universal_dev.iob2`, `data/universal_test_masked.iob2`
  - Format: 5-column IOB2 (`token_col=1, tag_col=2`)
  - Test set is **masked** (gold tags removed) — predictions are submitted to LearnIT for official scoring (`baseline_main_v1.py` lines 107-121)
- **CoNLL-2003** - Similar domain (Reuters newswire)
  - Files: `data/news_train.iob2`, `data/news_dev.iob2`, `data/news_test.iob2`
  - Format: 2-column IOB2 (`token_col=0, tag_col=1`)
- **WIESP-2022** (a.k.a. "astro") - Different domain (astrophysics paper abstracts)
  - Files: `data/astro_train.iob2`, `data/astro_dev.iob2`, `data/astro_test.iob2`
  - Format: 2-column IOB2 (`token_col=0, tag_col=1`); blocked by paragraph rather than sentence (`scripts/run_experiment.py` `_eval_unit`, lines 333-338)

**Label unification:** Cross-dataset overlap is normalised in `data.normalize_tag()` (`data.py` lines 23-35) and `UNIFY_MAP` (lines 12-20): WIESP `Person`/`Organization`/`Location` → `PER`/`ORG`/`LOC`; CoNLL `MISC` → `O`. Unified label set has 65 labels (`config.py` lines 4-39).

**Dataset integrity:** SHA-256 hashes of every loaded split are recorded per run in `config.json` snapshots (`src/experiment/config_loader.py` `collect_dataset_hashes` lines 108-118, written by `logging_io.write_config_snapshot`).

## Submission / Grading Integration

**LearnIT (ITU course platform):** Out-of-band integration. The baseline writes EWT test predictions to `outputs/ewt_test_predictions.iob2` (`baseline_main_v1.py` lines 111-116, `data.save_predictions` in `data.py` lines 257-273). The user manually uploads that file to LearnIT to obtain the official EWT test F1. No API integration.

---

*Integration audit: 2026-05-07*

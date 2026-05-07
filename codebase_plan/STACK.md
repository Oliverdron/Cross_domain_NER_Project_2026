# Technology Stack

**Analysis Date:** 2026-05-07

## Languages

**Primary:**
- Python 3.11 - Used for all training, data parsing, evaluation, and orchestration code (`baseline_main_v1.py`, `data.py`, `model.py`, `trainer.py`, `config.py`, `scripts/`, `src/`)

**Secondary:**
- Bash - Used for SLURM job scripts on ITU HPC (`hpc/install_env.job`, `hpc/train_iter.job`, `hpc/smoke_test.job`)
- YAML - Used for experiment configuration (`experiments/config_conll.yaml`, `experiments/config_astro.yaml`)
- Jupyter Notebook - Used for exploratory data analysis (`eda.ipynb`)

## Runtime

**Environment:**
- CPython 3.11 (pinned in `hpc/install_env.job` line 44: `conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.11`)
- CUDA 12.1 runtime on HPC (`module load CUDA/12.1.1` in `hpc/train_iter.job` line 46)
- PyTorch wheels built against `cu121` (`hpc/install_env.job` lines 57-58)

**Package Manager:**
- pip - Direct installs via `pip install --no-cache-dir` (see `hpc/install_env.job`)
- conda (Anaconda3 system module on HPC) - Creates the `ner` env from conda-forge channel
- Lockfile: Not present (only an unpinned `requirements.txt` exists; PyTorch is installed separately from a custom index URL)

## Frameworks

**Core:**
- PyTorch (`torch`) - Tensor backend, optimization, GPU/CPU device management. Used in `trainer.py` (AdamW, `torch.cuda.*`, `clip_grad_norm_`), `data.py` (DataLoader), `model.py`, and `baseline_main_v1.py`
- Hugging Face Transformers (`transformers`) - Pretrained model + tokenizer + LR scheduler (`AutoModelForTokenClassification`, `AutoTokenizer`, `DataCollatorForTokenClassification`, `get_linear_schedule_with_warmup`, `set_seed`). Used in `model.py`, `data.py`, `trainer.py`, `baseline_main_v1.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`
- Hugging Face Datasets (`datasets`) - In-memory `Dataset` objects + batched `.map()` for tokenisation. Used in `data.py` (`from datasets import Dataset`)

**Testing:**
- Not detected — no `pytest`, `unittest`, or test directories present. Validation is done via the `--debug` flag in `scripts/run_experiment.py` and `scripts/run_baselines.py` (uses 0.1% of data) and the `hpc/smoke_test.job` SLURM script.

**Build/Dev:**
- No build tooling. Project is run as plain Python scripts. No `setup.py`, `pyproject.toml`, or installable package.

## Key Dependencies

**Critical (declared in `requirements.txt`):**
- `torch` - Deep learning framework (unpinned in `requirements.txt`; installed from `https://download.pytorch.org/whl/cu121` on HPC per `hpc/install_env.job` line 58)
- `transformers` - Hugging Face model hub + tokenizers (unpinned)
- `seqeval` - Entity-level NER metrics (P/R/F1, classification report, `get_entities`). Used in `trainer.py` and `src/experiment/evaluation.py`
- `numpy<2` - **Pinned below 2.0** because `seqeval` and older `transformers` paths break on numpy 2.x (documented in `hpc/README_HPC.md` line 150 and `requirements.txt` line 4)
- `tqdm` - Training progress bars (`trainer.py`)
- `datasets` - Hugging Face Datasets library
- `scikit-learn` - Confusion matrix + token-level macro F1 (`src/experiment/evaluation.py` line 12: `from sklearn.metrics import confusion_matrix, f1_score as token_f1_score`)
- `pyyaml` - Experiment config parsing (`src/experiment/config_loader.py` line 15)
- `pandas` - Listed in `requirements.txt` (likely for `eda.ipynb`; not imported in any `.py` file in scope)
- `matplotlib` - Listed in `requirements.txt` (likely for `eda.ipynb`)

**Infrastructure:**
- `evaluate` - Installed on HPC only (`hpc/install_env.job` line 62) but **not in `requirements.txt`** and not imported anywhere in `.py` files. Likely a leftover from initial bootstrap.

## Configuration

**Environment:**
- CLI arguments via `argparse` for the baseline (`config.py` `get_args()`)
- YAML files via `pyyaml` for the iterative experiments (`experiments/config_conll.yaml`, `experiments/config_astro.yaml`); loaded into frozen dataclasses (`ExperimentConfig`, `SourceCfg`, `TargetCfg`, `EvalSetCfg` in `src/experiment/config_loader.py`)
- Environment variables (HPC only):
  - `HF_HOME=/home/olgy/.cache/huggingface` - Caches downloaded `bert-base-cased` weights between runs (`hpc/train_iter.job` line 52)
  - `TMPDIR=/home/olgy/tmp` - Conda/pip scratch space
  - `PYTHONUNBUFFERED=1` - Live SLURM logs
  - `TOKENIZERS_PARALLELISM=false` - Avoids HF tokenizer fork warning

**Build:**
- No build config. Scripts are invoked directly: `python scripts/run_experiment.py --config experiments/config_conll.yaml`

## Platform Requirements

**Development:**
- Python 3.11
- A CUDA-capable GPU is preferred but not required. Both `baseline_main_v1.py` (lines 34-37) and `scripts/run_experiment.py` (lines 121-123) gracefully fall back to CPU when `torch.cuda.is_available()` returns False.
- `bert-base-cased` (~440 MB) is downloaded from Hugging Face Hub on first run.

**Production / HPC:**
- ITU HPC (SLURM) - Documented in `hpc/README_HPC.md`
- Partition: `acltr` for real training (`hpc/train_iter.job` line 17), `scavenge` for env install / smoke test
- Resources per training job: 1 GPU (V100/A100/H100/L40s/RTX6000/A30 all acceptable per `train_iter.job` lines 18-21), 8 CPUs, 48 GB RAM, up to 2 days wall-clock
- Data files (`data/*.iob2`) must be present locally — there is no remote dataset download. Total data size ≈ 26 MB across nine `.iob2` files in `data/`.

---

*Stack analysis: 2026-05-07*

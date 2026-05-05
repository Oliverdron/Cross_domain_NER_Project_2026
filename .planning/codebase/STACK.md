# Technology Stack

## Summary

This is a Python-based NLP research project for cross-domain Named Entity Recognition (NER). It fine-tunes BERT-family transformer models using PyTorch and HuggingFace Transformers, evaluated with seqeval span-level F1. Training is designed to run on CUDA GPUs, with SLURM job scripts targeting an ITU HPC cluster (V100, CUDA 12.1).

**Analysis Date:** 2026-05-05

## Languages

**Primary:**
- Python 3.11 — all model, data, training, and experiment orchestration code

## Runtime

**Environment:**
- Python 3.11 (pinned in `hpc/install_env.job` via `conda create -n ner python=3.11`)

**Package Manager:**
- `pip` inside a conda-managed environment named `ner`
- Lockfile: not present (unpinned `requirements.txt`)

## Frameworks

**Core ML:**
- `torch` (PyTorch) — neural network training, GPU tensors, `AdamW`, gradient clipping, `DataLoader`
  - CUDA target: 12.1 (`cu121` wheels, installed via `https://download.pytorch.org/whl/cu121`)
  - Optimizer: `AdamW` from `torch.optim`
  - Scheduler: `get_linear_schedule_with_warmup` from `transformers`
- `transformers` (HuggingFace) — pretrained BERT loading, tokenizer, token classification head, checkpoint save/load
  - Key classes used: `AutoModelForTokenClassification`, `AutoTokenizer`, `DataCollatorForTokenClassification`
  - Default model: `google-bert/bert-base-cased`
- `datasets` (HuggingFace) — in-memory `Dataset` object used as the bridge between parsed IOB2 data and PyTorch `DataLoader`

**Evaluation:**
- `seqeval` — entity-span-level F1, precision, recall, per-type classification report, entity extraction (`get_entities`)
- `scikit-learn` — token-level confusion matrix (`confusion_matrix`), token-level macro F1 (`f1_score`)

**Data & Config:**
- `pyyaml` — YAML experiment config loading (`experiments/config_*.yaml`)
- `numpy < 2` — array ops (pinned below 2.x due to seqeval/transformers compatibility)
- `pandas` — available as a dependency; used in analysis notebooks
- `matplotlib` — plotting (notebooks and saved `.png`)
- `tqdm` — progress bars during training and evaluation loops

**Notebooks:**
- Jupyter (`eda.ipynb`, `trials.ipynb`) — exploratory data analysis and prototyping

## Key Dependencies

**Critical:**
- `torch` — entire training pipeline depends on this; must match CUDA 12.1 on HPC
- `transformers` — model architecture, tokenizer, checkpoint management
- `seqeval` — primary evaluation metric (span-level F1)
- `numpy<2` — pinned; seqeval and some transformers internals break on numpy 2.x
- `datasets` — HuggingFace Dataset used for batched tokenization via `.map()`

**Supporting:**
- `scikit-learn` — confusion matrix and secondary token-level metrics in `src/experiment/evaluation.py`
- `pyyaml` — config parsing in `src/experiment/config_loader.py`
- `tqdm` — training/eval progress display in `trainer.py`

## Configuration

**Experiment Config:**
- YAML files in `experiments/`: `config_conll.yaml`, `config_astro.yaml`
- Loaded via `src/experiment/config_loader.py` into frozen `ExperimentConfig` dataclass
- Key parameters: `model_name`, `tokenizer_name`, `max_seq_len`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_ratio`, `num_epochs`, `early_stopping_patience`, `seeds`, `block_size`

**Global Label Config:**
- `config.py` — defines the full unified `LABEL_LIST` (57 tags covering EWT/CoNLL/WIESP), `LABEL2ID`, `ID2LABEL`, and CLI arg parser

**Build:**
- No build system; plain Python scripts run directly
- No `setup.py`, `pyproject.toml`, or `setup.cfg`

## Platform Requirements

**Development:**
- Python 3.11
- GPU strongly preferred (code falls back to CPU with a warning)
- HuggingFace model cache: `~/.cache/huggingface` (or `$HF_HOME`)

**Production / HPC:**
- ITU HPC cluster (`hpc.itu.dk`)
- SLURM scheduler — job scripts in `hpc/`
- GPU: NVIDIA V100 32GB (`--gres=gpu:v100:1`)
- CUDA 12.1 (`module load CUDA/12.1.1`)
- Anaconda3 module for environment management (`module load Anaconda3`)
- 48 GB RAM, 8 CPU cores per training job
- Max job time: 2 days on `acltr` partition

---

*Stack analysis: 2026-05-05*

# STRUCTURE

## Summary
Cross-domain NER research project. Three datasets (EWT, CoNLL-2003, WIESP astrophysics) mapped to unified 65-label set. Core pipeline: iterative injection of target-domain examples into EWT-trained BERT model, with full artifact logging per seed × iteration × eval set.

## Directory Layout

```
Cross_domain_NER_Project_2026/
│
├── config.py                    ← Unified label set (65 labels), LABEL2ID/ID2LABEL maps,
│                                  and CLI argument parser (get_args)
│
├── data.py                      ← IOB2 parsing, label normalisation, tokenisation &
│                                  label alignment, DataLoader factory, prediction saver
│
├── model.py                     ← Builds BERT token-classification model via
│                                  AutoModelForTokenClassification
│
├── trainer.py                   ← Training loop (AdamW + linear warmup), per-epoch
│                                  JSONL logging, best-checkpoint saving, evaluate()
│
├── baseline_main_v1.py          ← Standalone entry point: trains on EWT, evaluates
│                                  cross-domain, saves EWT test predictions
│
├── requirements.txt             ← Python dependencies
│
├── data/                        ← Raw IOB2 datasets
│   ├── universal_train.iob2     ← EWT — English Web Text (5-column)
│   ├── universal_dev.iob2
│   ├── universal_test_masked.iob2
│   ├── news_train.iob2          ← CoNLL-2003 newswire (2-column)
│   ├── news_dev.iob2
│   ├── news_test.iob2
│   ├── astro_train.iob2         ← WIESP-2022 astrophysics abstracts (2-column, paragraph-blocked)
│   ├── astro_dev.iob2
│   └── astro_test.iob2
│
├── experiments/                 ← YAML configs for iterative injection experiments
│   ├── config_conll.yaml        ← EWT→CoNLL: step_size=150 sentences, 8 iterations, 3 seeds
│   └── config_astro.yaml        ← EWT→Astro: step_size=5 paragraphs, 8 iterations, 3 seeds
│
├── scripts/
│   ├── run_experiment.py        ← Iterative injection runner: reads YAML config, runs
│   │                              seeds × iterations, writes full artifact tree to runs/
│   └── run_baselines.py         ← Three single-domain baselines (ewt_only, conll_only,
│                                  astro_only) — same output schema as iterative runs
│
├── src/
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── config_loader.py     ← Frozen dataclasses (ExperimentConfig, SourceCfg,
│   │   │                          TargetCfg, EvalSetCfg), YAML loader, SHA-256 dataset
│   │   │                          hasher, git commit capture
│   │   ├── injection.py         ← Seed-deterministic injection pool builder and slicer:
│   │   │                          build_injection_pool, slice_for_iter, select_examples
│   │   ├── evaluation.py        ← full_evaluate: span F1, token accuracy/F1, BIO-collapsed
│   │   │                          confusion matrix, per-type P/R/F1, per-example JSONL records
│   │   └── logging_io.py        ← Atomic file writers (JSON, JSONL, text, CSV), summary.csv
│   │                              schema (SUMMARY_COLUMNS), directory helpers, config snapshot
│   └── helper/
│       └── helper_funcs.py      ← Utility functions: parse_iob2 (lightweight), jaccard_vocab
│
├── hpc/                         ← SLURM job scripts for ITU HPC cluster (partition: acltr, V100)
│   ├── install_env.job          ← One-time conda environment setup
│   ├── smoke_test.job           ← Quick sanity check job
│   ├── train_iter.job           ← Main training job; pass config via --export=ALL,CFG=<path>
│   └── README_HPC.md            ← HPC usage instructions
│
├── outputs/                     ← Default output dir for baseline_main_v1.py
│   ├── best_model/              ← Best BERT checkpoint (HuggingFace format)
│   └── ewt_test_predictions.iob2
│
├── runs/                        ← Output root for iterative and baseline experiments
│   └── <experiment_name>/
│       ├── config.json
│       ├── summary.csv          ← Append-only: one row per (seed × iteration × eval_set)
│       └── seeds/
│           └── seed_<N>/
│               ├── injection_order.json
│               └── iter_<KKK>/
│                   ├── meta.json
│                   ├── metrics.json
│                   ├── per_type_metrics.json
│                   ├── train_log.jsonl
│                   ├── added_target_ids.txt
│                   ├── predictions_<eval_set>.jsonl
│                   └── confusion_matrix_<eval_set>.csv
│
├── docs/
│   ├── analysis_plan.md
│   └── training_logging_plan.md
│
├── eda.ipynb
└── trials.ipynb
```

## Key Design Decisions

**Label unification.** All three datasets mapped to shared 65-label set. WIESP `Person/Organization/Location` collapse to `PER/ORG/LOC`; CoNLL `MISC` maps to `O`. Normalisation at tokenisation time in `data.py:normalize_tag`.

**Iterative injection.** `scripts/run_experiment.py` implements the core research loop: starting from pure EWT-trained model (iteration 0), each iteration adds `step_size` more target-domain examples to training mix. Model reinitialised from pretrained BERT checkpoint at every iteration — no warm-starting from previous iteration's weights.

**Deterministic injection ordering.** `src/experiment/injection.py:build_injection_pool` shuffles target training IDs with fixed seed and writes `injection_order.json` before any training begins.

**Evaluation separated from training.** `src/experiment/evaluation.py:full_evaluate` computes span F1, token accuracy, token F1, BIO-collapsed confusion matrix, per-entity-type P/R/F1, and per-example prediction records with per-token softmax probabilities.

**Atomic writes.** Every output file (`logging_io.py`) written to `.tmp` sibling then `os.replace`-d — crash never leaves partial files. `summary.csv` append-only with same pattern.

**Three eval sets per run.** Every iteration evaluates on `ewt_dev` (source-forgetting probe), `conll_test`, and `astro_test` — same `summary.csv` schema covers both iterative runs and baselines.

## Gaps / Unknowns
- `runs/` empty — no iterative experiment results yet
- Analysis/plotting scripts absent despite `docs/analysis_plan.md` specifying ~30 plots

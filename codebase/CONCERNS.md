# CONCERNS

## Summary
Project is pre-results (runs/ empty, model weights deleted). Several silent bugs exist that would corrupt outputs or produce misleading metrics. HPC scripts have hardcoded paths that break for other team members. Analysis/plotting layer is entirely absent despite detailed planning docs.

## Critical Issues

1. **Dead `early_stopping_metric` config field.** Both YAMLs set `early_stopping_metric: "target_dev_span_f1"` — parsed and stored in `ExperimentConfig` but never passed to or read by `trainer.train()`. Trainer always stops on micro dev F1. Broken semantic promise.

2. **No model weights exist.** `outputs/best_model/` has only `config.json`. Commit "Deleting the saving of the models" removed weights. Iterative pipeline also deletes checkpoints after reload — final model unrecoverable without retraining.

3. **`save_predictions` column corruption (data.py:270).** Hardcodes `parts[2] = pred_tag` — only safe for EWT's 5-column format. Applying to CoNLL/WIESP 2-column files corrupts output silently.

## Technical Debt

- All three HPC `.job` scripts hardcode `/home/olgy/` — other team members' jobs run in wrong directory silently.
- Duplicate `parse_iob2` in `src/helper/helper_funcs.py` (missing `id`, `raw_lines`, `unit` fields vs canonical `data.py` version). Helper version not imported anywhere but could be picked up accidentally.
- `requirements.txt` pins only `numpy<2`; `transformers`, `seqeval`, all others unpinned. `evaluate` library in `install_env.job` not in `requirements.txt`.
- `baseline_main_v1.py` imports `set_seed` from transformers (line 3) but never calls it — dead import that shadows `seed_everything` alias.
- `num_epochs: 300` with patience 8 is normally fine, but if early stopping misfires job could overtrain for hours on HPC.
- Commented-out first version of `jaccard_vocab` still in `helper_funcs.py`.

## Reproducibility Risks

- `set_seed()` never sets `torch.backends.cudnn.deterministic = True` — cuDNN non-determinism not suppressed.
- Tokenizer never saved alongside model weights; checkpoints immediately `shutil.rmtree`'d after reload — final model cannot be recovered without retraining.
- HuggingFace dataset cache only disabled when `return_truncation_count=True`; stale cache could be used silently after data file changes.

## Missing Pieces

- Zero analysis/plotting scripts exist despite `docs/analysis_plan.md` specifying ~30 plots and tables. `pandas`, `matplotlib`, `scikit-learn` are dependencies but used nowhere.
- `errors_<set>.jsonl` artifact (planned in `docs/training_logging_plan.md`) never written by `run_experiment.py`.
- No EWT test prediction file produced by iterative pipeline — only `baseline_main_v1.py` generates it. Best iterative model cannot be submitted to LearnIT without adding this step.
- No test suite — parsing logic, label normalization, and injection pool slicing all untested.

## Gaps / Unknowns

- Whether `batch_size: 32` + `max_seq_len: 512` fits in 16 GB V100 VRAM (no gradient accumulation fallback).
- Actual truncation rate for astro paragraphs at 512 tokens (unknown until a run completes).
- Whether `hpc/logs/` must exist before SLURM opens log file — potential job failure if directory absent.
- Whether `block_size: 30` in `config_conll.yaml` was intended to affect slicing (it does not — `slice_for_iter` ignores it).

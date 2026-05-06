# TESTING

## Summary
No automated test suite exists. Verification happens through structured experiment output artifacts (JSONL, CSV, JSON) written after every training run. Primary metric is seqeval span-level F1 (BIO-aware).

## Evaluation Methodology
- **Primary metric:** seqeval span-level F1 (BIO-aware), implemented in `trainer.py` and `src/experiment/evaluation.py`
- **Secondary metrics:** token accuracy, token macro F1 (BIO-collapsed), per-token softmax confidence
- Results written to JSONL, CSV, and JSON artifacts after each run

## Fast Feedback
- `--debug` mode uses 0.1% data, 1 seed — only fast-feedback mechanism available
- No unit tests, no integration tests

## Reproducibility Controls
- Seed control across all runs
- SHA-256 dataset hashes
- Git commit snapshots embedded in output artifacts
- Atomic file writes for output

## Coverage Gaps
- `normalize_tag()` — untested
- `parse_iob2()` — untested
- Label alignment in `make_tokenize_fn()` — untested
- All `src/experiment/` utilities — untested
- Zero `test_*.py` files exist in the project

## Gaps / Unknowns
- No CI/CD pipeline detected
- No benchmark comparison harness
- Unclear if evaluation scripts are run as part of any automated pipeline

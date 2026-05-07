# Testing Patterns

**Analysis Date:** 2026-05-07

## Test Framework

**Status: NO automated tests are committed to this repository.**

Searches performed at the project root for `test_*.py`, `*_test.py`, `tests.py`, `conftest.py`, `pytest.ini`, `tox.ini`, `pyproject.toml`, and any `tests/` directory all returned **zero results**. There is no `pytest`, `unittest`, `nose`, `hypothesis`, or `unittest.mock` import anywhere in the codebase.

The closest things to a test are:
- `hpc/smoke_test.job` — a SLURM job whose name suggests a smoke test for the HPC pipeline, not an automated unit/integration test.
- The `--debug` flag on the runner scripts (`scripts/run_experiment.py:109`, `scripts/run_baselines.py:278`) which slices each split to 0.1% and runs one seed. This is a manual pipeline-shakedown mode, not a regression test.
- The Jupyter notebook `eda.ipynb` at the repo root — exploratory data analysis, not a test harness.

**Runner:** None.
**Assertion Library:** None.
**Run Commands:** Not applicable.

## Test File Organization

**Location:**
- Not applicable — no test files exist.
- `.gitignore` does reserve `.pytest_cache/`, `.tox/`, `.nox/`, `.coverage`, `.coverage.*`, `htmlcov/`, `coverage.xml` (the standard Python ignore template) but no tools currently produce these artifacts.

**Naming:**
- N/A.

**Structure:**
- N/A.

## Test Structure

- N/A. The codebase relies on (1) eyeball verification of `print` output during runs, (2) the `--debug` reduced-data smoke run, and (3) post-hoc inspection of `summary.csv` / `metrics.json` artifacts in `runs/`.

## Mocking

- Not used. There is no `unittest.mock`, `pytest-mock`, `responses`, or `httpretty` import. Models, tokenizers, and HuggingFace `Dataset` objects are constructed for real even in the `--debug` path.

## Fixtures and Factories

- Not used in a testing sense. Reproducibility-related "fixtures" are real artifacts written to disk for audit:
  - `runs/<exp>/config.json` — frozen config + dataset SHA256s + git commit (`src/experiment/logging_io.py:130-150`)
  - `runs/<exp>/seeds/seed_<n>/injection_order.json` — deterministic target-injection pool (`src/experiment/injection.py:34-38`)
  - `runs/<exp>/seeds/seed_<n>/iter_<k>/added_target_ids.txt` — exactly what was added at iteration k

## Coverage

**Requirements:** None enforced. No `coverage.py`, no `.coveragerc`, no CI gate.

**View Coverage:** Not applicable.

## Test Types

- Unit Tests: None.
- Integration Tests: None.
- E2E Tests: None automated. The `--debug` mode is the de-facto end-to-end shakedown; it runs the real pipeline on 0.1% of data with one seed and a truncated schedule.
- Smoke Tests: `hpc/smoke_test.job` (SLURM, manual).

## Common Patterns

- N/A. No async testing, no error testing, no parametric testing exists.

---

# Notes on what would need testing for ML-research reliability

Because this is a research / experimental ML codebase, full unit-test coverage may not be the priority — but the following targeted checks would meaningfully reduce silent-correctness risk. Each item is paired with the file/function it would exercise.

## Tier 1 — Correctness-critical (recommend writing first)

1. **`data.normalize_tag`** (`data.py:23-35`)
   - Verify the four mapping rules: WIESP `Person/Organization/Location` → `PER/ORG/LOC`; any tag containing `MISC` → `O`; in-set tags pass through; out-of-set tags → `O`.
   - Risk if wrong: silently mislabels training data across all three datasets; F1 numbers become incomparable.

2. **`data.parse_iob2`** (`data.py:40-97`)
   - Verify column-index handling for both EWT (5-col, `token_col=1, tag_col=2`) and CoNLL/WIESP (2-col, `token_col=0, tag_col=1`).
   - Verify blank-line and `#`-comment delimiting, the `_flush` boundary, and the `id` field format `f"{stem}_{idx:05d}"`.
   - Verify the `ValueError` raised on unknown `unit`.
   - Risk if wrong: rows shifted off-by-one between datasets — model trains on garbage.

3. **`data.make_tokenize_fn` label alignment** (`data.py:173-219`)
   - Verify subword continuations and special tokens get `-100`, that the **first** subword of each word inherits the (normalised) word-level tag, and that `trunc_counter` increments only when word-level coverage is incomplete.
   - Risk if wrong: trainer ignores a fraction of every word's tokens or learns wrong tags on continuations.

4. **`trainer._decode_batch` and `_decode_batch_full`** (`trainer.py:24-67`)
   - Round-trip a known `(logits, labels)` tensor with `-100` markers and confirm the kept positions reproduce the original tag sequence after `argmax`. Confirm `_decode_batch_full` returns probabilities aligned with the kept labels.
   - Risk if wrong: F1 is computed on the wrong subset of positions.

5. **`trainer.set_seed`** (`trainer.py:17-21`)
   - Confirm two consecutive `set_seed(s)` calls produce identical `random`, `numpy.random`, and `torch.randn` draws.
   - Optional: extend the function to also pin `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, `os.environ["PYTHONHASHSEED"]`, and add a regression test that an end-to-end `train()` call on a tiny fixture produces identical `best_dev_f1` across two runs.
   - Risk if wrong: experiments are silently non-reproducible despite the seed.

6. **`src/experiment/injection.build_injection_pool`** (`src/experiment/injection.py:16-39`)
   - Same `seed` + same example `id` set must produce identical ordering AND identical `order_hash`.
   - Different seeds must produce different ordering with overwhelming probability.
   - Risk if wrong: cross-seed comparisons are invalid — what looks like seed variance is actually different example ordering.

## Tier 2 — Pipeline integrity

7. **`src/experiment/evaluation._bio_collapse` and `full_evaluate`** (`src/experiment/evaluation.py:17-89`)
   - Confirm `B-X`, `I-X` both collapse to `X`; `O` stays `O`. Confirm the per-example `predictions` records align 1:1 with the input `example_ids` and `example_tokens` (the docstring requires the dataloader not to be shuffled — a test would catch a future regression that shuffles eval data).

8. **`src/experiment/logging_io._atomic_write` and `append_summary_row`** (`src/experiment/logging_io.py:56-105`)
   - Verify a simulated mid-write crash leaves the prior file intact (write to `tmp`, do not call `os.replace`, confirm target file still has old contents).
   - Verify `append_summary_row` is idempotent on the header and produces N rows after N calls, with stable column order from `SUMMARY_COLUMNS`.

9. **`src/experiment/config_loader.load_config`** (`src/experiment/config_loader.py:69-97`)
   - Round-trip both `experiments/config_conll.yaml` and `experiments/config_astro.yaml` through `load_config` → `to_dict` and confirm shape stability.
   - Verify the `ExperimentConfig` dataclass is genuinely frozen (`FrozenInstanceError` on attribute assignment).

10. **`src/experiment/config_loader.sha256_file`** (`src/experiment/config_loader.py:100-105`)
    - Hash a known small file and compare to a precomputed digest, to lock in the chunked-read implementation.

## Tier 3 — Dataset-level invariants (cheap to add as runtime asserts or smoke tests)

11. After loading, every example dict must satisfy `len(ex["tokens"]) == len(ex["ner_tags"]) == len(ex["raw_lines"])` (data.parse_iob2 contract).
12. After `prepare_split`, every batch returned by `make_dataloader` must have `input_ids.shape == labels.shape`, and the count of `labels == -100` plus the count of `labels in LABEL2ID.values()` must equal `labels.numel()`.
13. `entity_density` (`data.py:100-115`) on an empty list returns `0.0` (currently true but not test-locked).

## Tier 4 — Numerical / metrics sanity

14. Construct a tiny gold sequence and an identical pred sequence and confirm `seqeval.f1_score` returns `1.0` and `classification_report` reports support correctly — protects against future seqeval API drift, since the project pins no version (`requirements.txt` has just `seqeval` with no constraint).
15. Confirm `build_summary_row` (`src/experiment/logging_io.py:169-205`) populates every column listed in `SUMMARY_COLUMNS` (`logging_io.py:39-51`); missing keys silently become `""` today, which is a foot-gun for downstream plotting.

## Tier 5 — Things explicitly NOT worth unit-testing

- `model.build_model` — a four-line call to `AutoModelForTokenClassification.from_pretrained`. Network-dependent, tested implicitly by every run.
- `trainer.train` and `trainer.evaluate` end-to-end loops — better covered by the existing `--debug` smoke run than by mocked unit tests.
- The `print` formatting in `print_results_table` (`baseline_main_v1.py:13-25`) — cosmetic.

## Suggested minimal testing setup

If/when tests are added, the lowest-friction path matching the project's existing style would be:

```
tests/
  test_normalize_tag.py
  test_parse_iob2.py
  test_label_alignment.py
  test_set_seed.py
  test_injection_pool.py
  test_atomic_writes.py
  test_config_loader.py
  fixtures/
    tiny.iob2          # 3-5 sentences, 2-col
    tiny_ewt.iob2      # 3-5 sentences, 5-col
```

Run via `pytest` from the repo root (the `sys.path` shim in `scripts/` already establishes the import-from-root convention; `pytest` handles this natively when run from the repo root). Add `pytest` to `requirements.txt` (currently absent) and consider committing a minimal `pyproject.toml` or `pytest.ini` setting `testpaths = ["tests"]` so the tooling slot reserved in `.gitignore` is actually used.

---

*Testing analysis: 2026-05-07*

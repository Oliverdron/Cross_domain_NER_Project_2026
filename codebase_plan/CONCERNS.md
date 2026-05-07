# Codebase Concerns

**Analysis Date:** 2026-05-07
**Working dir:** `/Users/samuel/Desktop/ITU/4th Semester/Natural Language and Deep Learning/Cross_domain_NER_Project_2026`

## Tech Debt

**Two parallel `parse_iob2` implementations:**
- Issue: `src/helper/helper_funcs.py` defines a stripped-down `parse_iob2` (no `id`, no `raw_lines`, no `unit`) that disagrees with the canonical `parse_iob2` in `data.py`. The helper module also has a duplicated `jaccard_vocab` — one is commented out (lines 20-23), the other shadows it.
- Files: `src/helper/helper_funcs.py:1-18` vs `data.py:40-97`
- Impact: Anything importing from `helper_funcs` gets a different example schema (no `id`), so `select_examples` / `build_injection_pool` would silently produce empty lists.
- Fix: Delete `src/helper/helper_funcs.py:parse_iob2` and re-export `data.parse_iob2`. Drop the commented `jaccard_vocab` block.

**`baseline_main_v1.py` is the legacy entry point:**
- Issue: The "v1" name plus the parallel `scripts/run_experiment.py` (YAML iterative runner) and `scripts/run_baselines.py` (train-once) implies `baseline_main_v1.py` is superseded. It still runs and produces an `outputs/best_model/` checkpoint nothing consumes.
- Files: `baseline_main_v1.py` (125 lines), `BASELINE_V1.md`
- Impact: Two ways to "train the baseline" with different defaults (`config.py` defaults `epochs=5`, no early stopping; `scripts/run_baselines.py` uses `epochs=100`, `early_stopping_patience=3`).
- Fix: Remove or move under `legacy/`; update `BASELINE_V1.md`.

**`BASELINE_V1.md` is stale:**
- Issue: References an entry point named `main.py` that does not exist (actual file is `baseline_main_v1.py`). No mention of `scripts/run_experiment.py`, YAML configs, or the iterative-injection workflow.
- Files: `BASELINE_V1.md:5-13, 32-37`
- Fix: Rewrite or replace with a top-level `README.md` describing both runners.

**Dead/unused imports:**
- Issue: `baseline_main_v1.py:3` imports `set_seed` from `transformers` and never uses it (uses local `seed_everything`). Imports `precision_score, recall_score, f1_score` (line 9) and never uses them.
- Files: `baseline_main_v1.py:3, 9`
- Fix: Trim imports.

**Committed `.DS_Store` files survive in tree:**
- Issue: Despite `.gitignore` listing `.DS_Store` and recent commit `4d2f562 "Untrack .DS_Store file"`, working-tree `.DS_Store` files exist in `runs/`, `src/`, `src/experiment/` and at the repo root.
- Files: `./.DS_Store`, `runs/.DS_Store`, `src/.DS_Store`, `src/experiment/.DS_Store`
- Fix: `find . -name .DS_Store -delete`.

**Mixed Python-version `__pycache__/` directories:**
- Issue: Working tree contains both `__pycache__/data.cpython-310.pyc` (last touched 2025-04-23) and `__pycache__/data.cpython-312.pyc`. HPC env pins `python=3.11` (`hpc/install_env.job:44`).
- Files: `__pycache__/`, `src/experiment/__pycache__/`
- Impact: Three Python versions in play (3.10 local stale, 3.11 HPC, 3.12 local). Reproducibility risk.
- Fix: `find . -name __pycache__ -exec rm -rf {} +`. Pin a `.python-version`.

**`requirements.txt` has zero version pins:**
- Issue: All ten deps are unpinned (`torch`, `transformers`, `seqeval`, `tqdm`, `datasets`, `scikit-learn`, `pyyaml`, `pandas`, `matplotlib`); only `numpy<2` is constrained.
- Files: `requirements.txt`
- Impact: A re-install months from now will yield different F1 scores, making the committed `git_commit` snapshot meaningless without a lockfile.
- Fix: Generate a lockfile from the HPC env and commit it.

**`tokenizer.name_or_path` is the only model-version anchor:**
- Issue: `write_config_snapshot` records the tokenizer name and `transformers_version` but NOT the model's HuggingFace Hub revision. `google-bert/bert-base-cased` weights can change.
- Files: `src/experiment/logging_io.py:130-150`, `model.py:12-19`
- Fix: Pin `revision="..."` in `from_pretrained`; record it in the snapshot.

## Known Bugs

**`save_predictions` assumes EWT 5-column format unconditionally:**
- Symptoms: `data.py:269` does `parts[2] = pred_tag`, only correct for EWT. Calling on CoNLL/astro (2 columns) raises `IndexError`. Currently only `baseline_main_v1.py:112` uses it (for EWT) — latent.
- Files: `data.py:257-273`
- Trigger: Future caller passing CoNLL/astro sentences.
- Fix: Pass `tag_col` into `save_predictions`, or store it on each example.

**`save_predictions` crashes when `output_path` has no directory component:**
- Symptoms: `data.py:263` does `os.makedirs(os.path.dirname(output_path), exist_ok=True)`. If `output_path` is a bare filename, `os.path.dirname` returns `""` and `os.makedirs("")` raises `FileNotFoundError`.
- Files: `data.py:263`
- Fix: `os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)` (the same idiom is used correctly in `logging_io._atomic_write` at `src/experiment/logging_io.py:57`).

**`run_experiment.py` does not reseed inside the iteration loop:**
- Symptoms: `set_seed(seed)` is called once per seed at `scripts/run_experiment.py:195`. The iteration loop (line 203) then runs many fine-tunes back-to-back; each iteration consumes RNG state from the previous. Same seed but different `schedule[]` ordering produces different iter-0 results.
- Files: `scripts/run_experiment.py:193-205`
- Trigger: Always (by design).
- Fix: Reseed at the start of every iteration (`seed_everything(seed + k)` or document explicitly).

**`logging_io.append_summary_row` is O(N²) over rows:**
- Symptoms: Each call reads the entire existing CSV, appends one row, and atomic-renames. For ~180 rows per config it's fine; larger sweeps degrade.
- Files: `src/experiment/logging_io.py:83-105`
- Fix: Open in append mode; only write a header on first call.

## Security Considerations

**Personal email + ITU username in HPC scripts:**
- Risk: `hpc/train_iter.job:28` hardcodes `--mail-user=olgy@itu.dk`; three SBATCH files reference `/home/olgy/...`; `hpc/README_HPC.md:36` references the `Oliverdron` GitHub user. Public-record account names — not secrets.
- Files: `hpc/train_iter.job:28,36-52`, `hpc/install_env.job:26-28`, `hpc/smoke_test.job:25-27,38`, `hpc/README_HPC.md:12-13,29,35-36,53,70,96,106,110,116,149`
- Fix: Replace with `${USER}` and `--mail-user=${SLURM_JOB_USER:-$USER}@itu.dk`. Low priority.

**No `.env` file, no API keys, no committed secrets:**
- Risk: None detected. `.gitignore` covers `.env`, `.envrc`, `.venv`. Repo contains no `.env*`, no `*.pem`, no `id_rsa`, no `serviceAccountKey.json`.
- Recommendation: Continue not adding secrets.

**`subprocess.run(["git", "rev-parse", "HEAD"], ...)` silently swallows errors:**
- Risk: `src/experiment/config_loader.py:121-129` shells out to git; bare `except Exception` returns `"unknown"`.
- Fix: Log the error so a missing git binary isn't invisible in `runs/<exp>/config.json`.

## Performance Bottlenecks

**`prepare_split` re-tokenises the source training set on every iteration:**
- Problem: `scripts/run_experiment.py:214` calls `prepare_split(mix, ...)` for every `(seed, k)`, where `mix = source_train + target_chunk`. The source portion is identical across iterations — ~20K EWT sentences × 60 iterations = 1.2M redundant tokenisations per config.
- Files: `scripts/run_experiment.py:209-220`, `data.py:222-245`
- Fix: Pre-tokenise `source_train` and the full target pool once; per-iteration only `concatenate_datasets([src_tok, tgt_tok.select(target_chunk_ids)])`.

**`load_from_cache_file=False` in `prepare_split`:**
- Problem: HuggingFace `Dataset.map` cache is explicitly disabled at `data.py:241`.
- Fix: Hash `(LABEL_LIST, UNIFY_MAP, max_length, tokenizer.name_or_path)` into the cache key and re-enable.

**`shutil.rmtree(train_result["best_model_dir"])` after every iteration:**
- Problem: `scripts/run_experiment.py:244` and `scripts/run_baselines.py:210` save the best HF model to disk, immediately reload it, then delete it. ~440 MB write+read per iteration × 60 iterations = ~25 GB redundant disk I/O per config.
- Fix: Refactor `trainer.train()` to keep the best `state_dict` in CPU memory and return the loaded model.

**No `num_workers` on `DataLoader`:**
- Problem: `data.py:251` uses default `num_workers=0`. HPC nodes have `--cpus-per-task=8` — seven cores idle.
- Fix: Pass `num_workers=4, pin_memory=True`; seed a `generator` for determinism.

**No mixed precision / `bf16`:**
- Problem: Training runs in fp32 (`trainer.py:185-211`). On V100/A100, `torch.cuda.amp.autocast` would roughly halve memory and ~1.5× throughput.
- Fix: Wrap forward/backward in `autocast` + `GradScaler`.

**`token_f1_score` rebuilds confusion matrix that was already computed:**
- Problem: `src/experiment/evaluation.py:50-56` computes the BIO-collapsed CM once via `sklearn.confusion_matrix` then calls `sklearn.f1_score` on the same flat lists, which internally rebuilds the matrix.
- Fix: Compute macro F1 from the existing matrix (diagonal vs row/col sums). Negligible runtime impact at current scale.

## Fragile Areas

**Determinism stops at PyTorch ops; no `cudnn.deterministic` flag:**
- Files: `trainer.py:17-22`
- Why fragile: `set_seed` covers `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all` — but NOT `torch.backends.cudnn.deterministic = True` or `torch.use_deterministic_algorithms(True)`. cuDNN's autotuner picks different conv kernels across runs, yielding F1 differences of ~0.001-0.005 at the same seed.
- Test coverage: Zero — no automated test verifies same-seed reproducibility.

**`eval_examples` ordering is the contract that holds prediction JSONL alignment:**
- Files: `scripts/run_experiment.py:165-174`, `src/experiment/evaluation.py:60-73`
- Why fragile: `full_evaluate` zips `example_ids`, `example_tokens`, `gold_seqs`, `pred_seqs` by index. If an eval `DataLoader` ever uses `shuffle=True`, all per-example records become silently misaligned. There's no integrity check.
- Safe modification: Always build eval loaders with `shuffle=False` (currently honoured at `run_experiment.py:82` and `run_baselines.py:114`).

**`raw_lines` writes to `parts[2]` without bounds check:**
- Files: `data.py:60-94, 257-273`
- Why fragile: `save_predictions` does `parts = raw_line.split("\t"); parts[2] = pred_tag` with no `assert len(parts) >= 3`.
- Fix: Add the assertion.

**Unit-of-injection mismatch between datasets:**
- Files: `experiments/config_astro.yaml:8`, `experiments/config_conll.yaml:7`, `experiments/config_astro.yaml:33`, `experiments/config_conll.yaml:30`
- Why fragile: The astro config uses `unit: paragraph` (~5 EWT sentences ≈ 1 paragraph). The astro schedule (`5, 7, 10, ...`) is paragraphs; CoNLL schedule (`150, 200, ...`) is sentences. Cross-dataset comparisons "at the same iter" are not apples-to-apples. Comments warn but no code enforces or normalises.
- Fix: Add a derived `n_target_tokens` column to `summary.csv`.

**No validation that source/target/eval YAML paths exist before training starts:**
- Files: `src/experiment/config_loader.py:69-118`
- Why fragile: A typo'd path inside `eval_sets` only fails when `parse_iob2` runs — possibly partway through HPC walltime. `collect_dataset_hashes` silently skips missing files (`if os.path.exists(p)` at line 118).
- Fix: Assert every configured path resolves.

**`config.py` hardcodes the 65-label `LABEL_LIST` — single source of truth, no enforcement:**
- Files: `config.py:4-39`, `data.py:23-35`
- Why fragile: A new entity type silently maps to `"O"` via `normalize_tag`'s fall-through (`return tag if tag in LABEL2ID else "O"`). Invisible label loss.
- Fix: At parse time, log distinct tags encountered; warn when any are remapped to `"O"` outside the documented `MISC` rule.
- Test coverage: None.

## ML-Specific Concerns

**EWT test set is fully masked (all `O`):**
- Verified by `awk` over `data/universal_test_masked.iob2`: only one tag value present (`O`, 25,097 rows). The masked test cannot be evaluated locally — predictions must be uploaded to LearnIT.
- Files: `data/universal_test_masked.iob2`, handled at `data.py:150` and `experiments/config_conll.yaml:45`
- Risk: A future contributor not knowing this could compute meaningless local F1 on the masked test. Currently the configs route around it by using `ewt_dev` as the source-forgetting probe.
- Mitigation: Comment in YAML; not enforced in code.
- Fix: Add an assertion that the EWT test set's tag column is exclusively `O` and warn if anyone tries to compute F1 on it.

**Label leakage / contamination — manual disjointness:**
- `scripts/run_experiment.py:160-163` builds `target_dev_loader` from `cfg.target.dev` for early stopping. Currently `cfg.target.dev = news_dev.iob2` and `eval_sets.conll_test = news_test.iob2` (`experiments/config_conll.yaml:38-50`) — distinct files, no contamination found.
- However, NOTHING in the code asserts that `cfg.target.dev` is disjoint from any `cfg.eval_sets[*].path`. A future YAML typo could route `eval_sets.conll_test` to `news_dev.iob2`, silently double-counting the dev set as both selector and reporter.
- Fix: Assert `{target.dev path} ∩ {eval_sets paths} == ∅` after `load_config`.

**No cross-domain example overlap audit:**
- Problem: No code verifies EWT/CoNLL/WIESP have disjoint sentences. Identical or near-duplicate sentences appearing in EWT train and CoNLL test would inflate cross-domain F1. The `jaccard_vocab` helper (`src/helper/helper_funcs.py:26-35`) only computes vocabulary overlap, not example overlap.
- Fix: Add a script that hashes `tuple(tokens)` per example and reports collisions across the six (train, dev, test) × 3 dataset splits.

**Label normalisation drops CoNLL `MISC` silently:**
- Files: `data.py:23-35`
- Behaviour: `MISC` → `O`, then any remaining unknown tag → `O`. This is a deliberate cross-domain unification choice but means CoNLL's `MISC` entities (a non-trivial fraction) become "background" during training and evaluation. Reported F1 is on the unified label set, NOT on CoNLL's original 4-type set — so direct comparison with published CoNLL-2003 baselines is invalid.
- Risk: External reviewers may misread the F1 numbers as comparable to CoNLL leaderboard.
- Mitigation: Documented in `BASELINE_V1.md:99`.
- Fix: Make the MISC-drop opt-in via a config flag; report CoNLL metrics computed on the original 4-type set in addition to the unified set.

## Scaling Limits

**`runs/` directory growth:**
- Predictions JSONLs are gitignored (`runs/**/predictions_*.jsonl`); other artifacts (`metrics.json`, `confusion_matrix_*.csv`, `train_log.jsonl`, `summary.csv`) ARE committed.
- Per-iter ~50 KB × 60 iters × 2 configs ≈ 6 MB per full sweep. Manageable.
- Path: Add `runs/**/*.csv` to gitignore if multi-config sweeps grow.

**Single-GPU training only:**
- bert-base on V100 fits batch_size=32, max_seq_len=512 (`hpc/README_HPC.md:124-128`).
- Larger backbones (`bert-large`, RoBERTa-large) require gradient accumulation (not implemented) or DDP (not implemented).
- Fix: Add `--gradient_accumulation_steps` to `trainer.train()`; introduce `accelerate` for multi-GPU.

## Dependencies at Risk

**`numpy<2` constraint:**
- Risk: Hard upper-bound because seqeval and older transformers paths break under numpy 2 (`hpc/README_HPC.md:150-151`).
- Migration: Track seqeval upstream; drop pin once a numpy-2-compatible release lands.

**`google-bert/bert-base-cased` is an external HuggingFace asset:**
- Risk: HuggingFace can rate-limit, rename, or remove model weights.
- Migration: Mirror weights into project-controlled storage; document fallback in `hpc/README_HPC.md`.

**`evaluate` library installed but never imported:**
- `hpc/install_env.job:62` installs `evaluate`; no source file imports it.
- Fix: Remove from the install command.

## Missing Critical Features

**No automated tests:**
- Zero `test_*.py` or `*_test.py` files. No `pytest`/`unittest` config. No CI.
- Blocks: Refactoring confidence; reproducibility verification; catching regressions on dependency bumps.

**No reproducibility regression check:**
- No script that reruns a small `--debug` sweep and asserts F1 falls within ε of a stored gold value.

**No analysis/plotting code committed:**
- `summary.csv` is the experiment's primary output but no committed code reads it. Plots presumably live in `eda.ipynb` (176 KB notebook) or a side notebook.
- Blocks: Reviewer reproducibility — they get a CSV and must write the analysis themselves.

**No mechanism to resume an interrupted iteration sweep:**
- If `train_iter.job` hits the 2-day SLURM wall partway through iter 17 of 20, there is no resume — the next submission starts at iter 0.
- Fix: Skip iter `k` if `runs/<exp>/seeds/seed_<S>/iter_<K:03d>/metrics.json` already exists.

## Test Coverage Gaps

**Untested: `parse_iob2` column-handling logic:**
- Not tested: 5-column EWT vs 2-column CoNLL/astro switching, blank-line splitting, `#` comment skipping, paragraph-vs-sentence unit propagation.
- Files: `data.py:40-97`
- Risk: A regression misaligns labels — only signal is degraded F1, no error.
- Priority: High.

**Untested: `normalize_tag` rules:**
- Not tested: WIESP→PER/ORG/LOC remapping, MISC→O, unknown tag fallthrough.
- Files: `data.py:23-35`
- Risk: Silent label corruption on dataset additions.
- Priority: High.

**Untested: Tokenisation/label alignment with subwords:**
- Not tested: That `-100` is correctly assigned to subword continuations / specials, that the first subword carries the original label, that truncation increments `trunc_counter`.
- Files: `data.py:184-219`
- Risk: A miscount shifts F1 directly.
- Priority: High.

**Untested: Injection determinism:**
- Not tested: Same seed → same `pool_ids` ordering across runs and Python versions.
- Files: `src/experiment/injection.py:16-39`
- Priority: Medium.

**Untested: `full_evaluate` ordering invariant:**
- Not tested: That `example_ids[i]` corresponds to the i-th batch of the unshuffled DataLoader.
- Files: `src/experiment/evaluation.py:31-89`
- Priority: High.

**Untested: `summary.csv` schema stability:**
- Not tested: That `SUMMARY_COLUMNS` and `build_summary_row` produce a row matching the header exactly.
- Files: `src/experiment/logging_io.py:39-51, 169-205`
- Priority: Medium.

---

*Concerns audit: 2026-05-07*

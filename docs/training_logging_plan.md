# Iterative Cross-Domain NER — Training Outputs & Logging Plan

## Goal of the experiment
Train on a source distribution, then iteratively inject more target-domain data, retraining and evaluating each step. We want to answer:
1. **How much target data is enough?** (the "data efficiency curve")
2. **Where does target data help vs. hurt?** (per entity type, per domain)
3. **Does adding target data cause source-domain forgetting?**
4. **Is improvement smooth or stepwise?** (i.e., do certain examples matter much more than others?)

To answer all of these from one set of runs, we need to save a lot — but most of it is cheap (JSON/CSV) and saves us re-running expensive experiments.

---

## Recommended directory layout

```
runs/
  exp_<name>/
    config.json                  # full hyperparams + git commit
    summary.csv                  # one row per iteration (the master table)
    iter_000/
      meta.json                  # iter info: target_n, seed, timestamp, runtime
      train_log.jsonl            # per-step or per-epoch loss/lr/grad_norm
      metrics.json               # all eval metrics on every test set
      per_type_metrics.json      # P/R/F1 per entity type, per eval set
      confusion_matrix_<set>.csv # token-level confusion, one per eval set
      predictions_<set>.jsonl    # raw predictions on each eval set
      errors_<set>.jsonl         # filtered to wrong entities (for fast inspection)
      added_target_ids.txt       # IDs of target examples used this iter
      checkpoint/                # model weights (or just final state_dict.pt)
    iter_001/ ...
    seeds/                       # if running multiple seeds, mirror the above
      seed_42/iter_000/...
```

The single most useful artifact is `summary.csv` — one row per (iteration × seed × eval_set) — because every plot in the report can be made from it.

---

## What to save — by category

### 1. Run-level metadata (once per experiment)
- Git commit hash, date, machine, GPU
- Full config: model name, batch size, LR schedule, max epochs, early-stopping criteria, optimizer, max_seq_len, label set
- Tokenizer name + version (drift here breaks reproducibility)
- Source dataset size, target dataset size, label inventory
- The exact pool of target examples that *could* be added (so the order/sampling is reproducible)
- A frozen copy of the dev/test splits (or hashes of them)

### 2. Per-iteration metadata
- `iteration_index` (0, 1, 2, …)
- `n_target_added_this_iter`, `n_target_total_so_far`, `n_source`, `target_fraction = n_target / (n_source + n_target)`
- `target_example_ids_added` (so you can do active-learning analyses later)
- `seed`
- Wall-clock training time, peak GPU memory
- Number of epochs actually run (if early stopping)
- Best-checkpoint epoch

### 3. Training dynamics (during fit)
Save per epoch (or per N steps) — a JSONL is easiest:
- `train_loss`, `dev_loss`
- `train_f1`, `dev_f1` (cheap if you eval each epoch)
- Learning rate at that step
- Gradient norm (catches instability)
- Optional: per-label loss if you can get it cheaply

This lets you plot training curves and detect overfitting per iteration — important because adding target data often changes the optimal stopping point.

### 4. Evaluation metrics (the headline numbers)
Run eval on **every** test set every iteration, not just target. At minimum:
- `source_test` (= news or universal — whichever is source)
- `target_test` (the held-out target set — astro?)
- `target_dev` (during training)
- `universal_test` (general benchmark / sanity)
- Optionally a **mixed test** that's stratified across all three

For each eval set, save:
- **Entity-level** (the standard NER metric, what `seqeval` gives you):
  - Precision, Recall, F1 — micro, macro, weighted
- **Token-level**:
  - Accuracy, P/R/F1 (less strict — useful for diagnosing "model got the type right but boundary wrong")
- **Boundary vs. type errors split out**:
  - Exact-match F1 vs. type-only F1 vs. boundary-only F1 (Partial-match metrics, MUC-style if you want to be thorough)
- Loss on the eval set
- Sample count (so the macro/micro distinction is interpretable)

### 5. Per-entity-type breakdown
This is where the most interesting findings usually live. For each eval set:
- P / R / F1 / support per entity type (PER, ORG, LOC, plus any astro-specific types)
- Number of predicted vs. gold spans per type
- This lets you make heatmaps of "F1 per type × iteration" — usually the most compelling plot in a cross-domain paper

### 6. Confusion data
- **Token-level confusion matrix** per eval set (BIO-collapsed: i.e., merge B-PER and I-PER into PER) — save as CSV so it loads instantly
- Optional: full BIO confusion (catches IOB-tagging errors)
- **Span-level confusion**: for each gold span, what did the model predict (correct / wrong type / partial / missed)? Counts are enough; predictions JSONL covers the details.

### 7. Raw predictions (the goldmine)
For each eval set, one JSONL line per sentence:
```json
{
  "id": "astro_test_0042",
  "tokens": [...],
  "gold_tags": [...],
  "pred_tags": [...],
  "pred_logits_or_probs": [...],   // optional but very useful
  "gold_spans": [{"type":"PER","start":3,"end":5,"text":"..."}],
  "pred_spans": [...]
}
```
With this you can recompute *any* metric retroactively, do error analysis, build qualitative example tables, and run significance tests — without rerunning training.

If logits are too big, save **per-token max prob** + **predicted label** instead. That's enough for confidence/calibration analyses.

### 8. Confidence & calibration
Cheap to add and very useful:
- Mean predicted probability for correct vs. incorrect predictions
- Reliability diagram data (binned confidence vs. accuracy) — save the bins
- Mean entropy per token / per sentence
- Optional: **Expected Calibration Error** (ECE)

This lets you ask "does adding target data make the model *correctly* more confident, or just more confident?"

### 9. Cross-domain-specific diagnostics
- **Forgetting score**: source_F1 at iter_k minus source_F1 at iter_0 (negative = forgetting)
- **Transfer score**: target_F1 at iter_k minus target-only baseline trained from scratch on the same target data
- **OOV rate**: fraction of target test tokens unseen in current training mix
- **Vocabulary overlap** (Jaccard — you already have this) between current training mix and target test
- Per-type **support shift**: how the entity-type distribution of training data changes with each iteration

### 10. Model artifacts
- Best checkpoint per iteration (or just `state_dict.pt` — full optimizer state is rarely needed afterward)
- A small `model_card.json`: param count, model class, base encoder
- Hash of the trained weights (for reproducibility checks)

### 11. Robustness / statistical reliability
**Run multiple seeds per iteration** (3 minimum, 5 ideal). Without this, you can't tell whether a bump in F1 is real or noise. Save:
- Per-seed metrics (so you can plot mean ± std bands)
- Bootstrap CIs on the test set if you have time

--- 

## The master `summary.csv`

One row per (experiment, seed, iteration, eval_set). Columns:

```
exp_name, seed, iteration, n_target_total, target_fraction,
eval_set, entity_f1_micro, entity_f1_macro, entity_precision, entity_recall,
token_f1, token_acc, eval_loss,
f1_PER, f1_ORG, f1_LOC, f1_<astro_types>...,
forgetting_vs_iter0, ece, mean_confidence,
train_time_sec, best_epoch
```

Almost every plot below comes from grouping/pivoting this one file.

---

## Plots & analyses this enables

Once the data above is saved, your friend (or you) can produce all of these without rerunning anything:

1. **Data-efficiency curve** — target_F1 vs. n_target_total, with mean ± std band over seeds. *The headline figure.*
2. **Source vs. target F1 over iterations** — shows the trade-off / forgetting.
3. **Per-entity-type heatmap** — entity types (rows) × iteration (cols), cell = F1. Reveals which types need the most target data.
4. **Confusion matrix at iter 0 vs. iter N** — side-by-side, shows which mistakes get fixed.
5. **Training curves per iteration** — overlaid loss/F1 trajectories, shows when the model needs more or fewer epochs.
6. **Calibration / reliability diagrams** at iter 0 vs. iter N.
7. **Marginal value plot** — ΔF1 per added example bucket. Tells you if returns are diminishing or if there's a sharp threshold.
8. **Significance / variance plot** — error bars over seeds at each iteration; useful for an "is this difference real?" answer.
9. **Error category breakdown** — stacked bar of (correct / wrong-type / wrong-boundary / missed / spurious) per iteration.
10. **OOV rate vs. F1 scatter** — connects vocabulary coverage to performance.
11. **Qualitative table** — pick 5–10 sentences where iter_0 was wrong and iter_N is right, and vice versa.
12. **Cost / benefit plot** — F1 gain per minute of training, per labeled example, etc.

---

## Practical tips

- **JSONL > pickle** for predictions — you'll thank yourself when you want to grep them.
- **Log eagerly, decide later.** Disk is cheap; rerunning a sweep is not.
- **Hash the data splits** and store the hash in config — protects against silently changing the test set.
- Use a logging library (Weights & Biases, MLflow, or even just TensorBoard) **on top of** these files, not instead of them. Files are durable; cloud experiments come and go.
- Make `summary.csv` append-only after each iteration so a crashed run still leaves a usable partial result.

---

## Minimum viable version (if time is short)
If your friend wants to start small, prioritize in this order:
1. `summary.csv` with iteration, n_target, eval_set, F1 (micro), F1 per type — covers ~70% of plots
2. Predictions JSONL on the target test set — enables retroactive error analysis
3. Multiple seeds — without this no plot can be trusted
4. Source-set evaluation each iteration — needed for the forgetting story
5. Everything else

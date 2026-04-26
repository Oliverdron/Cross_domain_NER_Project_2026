# Analysis Plan — Iterative Cross-Domain NER

Companion to `training_logging_plan.md`. This is the menu of analyses to run *after* the iterative training is complete. Organized by research question, with plot / table / statistic suggestions for each.

The four research questions we want to answer:
1. **How much target data is enough?** (the data-efficiency / saturation question)
2. **Where does target data help vs. hurt?** (per type, per domain, per example)
3. **Is the source domain forgotten?** (interference / catastrophic forgetting)
4. **Are improvements real or noise?** (statistical reliability)

---

## A. Headline / "money" plots
These are what you'd put in the abstract or conclusion.

### A1. Data-efficiency curve  *(the single most important plot)*
- **X**: number of target examples added (or target_fraction)
- **Y**: target-test F1 (entity-level micro)
- **Lines**: mean across seeds, shaded band = ±1 std or 95% bootstrap CI
- **Reference lines**:
  - iter_0 = source-only baseline (horizontal dashed)
  - target-only baseline (train from scratch on target only) (horizontal dotted)
  - "upper bound" = train on full source + full target (horizontal solid)
- **What to look for**: where does the curve plateau? How much target data closes 50%, 80%, 95% of the gap to the upper bound?

### A2. Source vs. target trade-off curve
- Same X-axis, two Y-curves: source-test F1 and target-test F1
- Shows whether gains on target come at the cost of source performance
- Annotate the "sweet spot" iteration: max(target_F1) subject to source_F1 ≥ threshold

### A3. Saturation table
A small table reporting:
- F1 at 0%, 10%, 25%, 50%, 100% of target data
- The "knee" — first iteration where ΔF1 < ε for k consecutive iterations
- Number of target examples needed to reach 90% / 95% / 99% of upper-bound F1

---

## B. Per-entity-type analyses
This is where the *interesting* paragraph of the report lives.

### B1. Per-type F1 heatmap
- Rows: entity types (PER, ORG, LOC, plus astro-specific types)
- Cols: iteration index
- Cell color: F1
- Reveals which types start poor and recover, which start fine, which never improve

### B2. Per-type data-efficiency curves (small multiples)
- One mini line plot per entity type, all on a shared grid
- Same axes as A1, but Y-axis is per-type F1
- Tells you if some types saturate at 50 examples while others need 1000

### B3. Type-frequency vs. type-improvement scatter
- X: support of the entity type in the target training pool
- Y: F1 improvement from iter_0 to iter_N
- Tests the hypothesis "more frequent types improve faster"

### B4. Type-rarity in source vs. improvement
- For each type: log(count_in_source) vs. F1_lift_from_target_data
- A clean negative correlation = "target data helps most for types missing from source"

---

## C. Confusion / error-structure analyses

### C1. Confusion matrix at iter_0 vs. iter_N (side by side)
- Token-level, BIO-collapsed (merge B-PER/I-PER → PER)
- Normalize by row (= recall) and by column (= precision) — show both
- Off-diagonal mass = the systematic mistakes

### C2. **Confusion-matrix delta** (very useful, often skipped)
- Cell value = (count at iter_N) − (count at iter_0)
- Negative diagonal = better; negative off-diagonal = fewer of that mistake; positive off-diagonal = a *new* mistake the model has learned (worth investigating)

### C3. Error-category stacked bar over iterations
For each iteration, decompose target-test errors into:
- Correct
- Wrong type (right boundary, wrong label)
- Wrong boundary (right label, off-by-one or partial overlap)
- Missed (gold span, no prediction)
- Spurious (predicted span, no gold)

Stacked-bar over iterations shows *what kind* of errors target data fixes. (Often: target data fixes "missed" before it fixes "wrong type".)

### C4. Per-domain confusion difference
- Confusion matrix on source-test at iter_N minus the same at iter_0
- Negative diagonal here = forgetting; quantify *which* types are forgotten

---

## D. Forgetting / interference analyses

### D1. Forgetting curve
- X: iteration
- Y: source_test_F1 minus source_test_F1 at iter_0
- Negative values = forgetting. Plot per entity type as small multiples too.

### D2. Backward Transfer (BWT) scalar
- Standard continual-learning metric: BWT = mean over types of (final source F1 − initial source F1)
- Report a single number alongside final target F1

### D3. Forward Transfer (FWT) scalar
- For each iteration's added target chunk, F1 on it *before* it was added vs. *after*
- Captures "how well does the source-trained model already generalize to target?"

---

## E. Confidence & calibration

### E1. Reliability diagrams (iter_0 vs. iter_N)
- X: predicted probability bin
- Y: empirical accuracy in that bin
- Diagonal = perfectly calibrated
- Useful to show: does adding target data make the model better-calibrated, or just more confidently wrong?

### E2. ECE (Expected Calibration Error) over iterations
- Single line plot, X = iteration, Y = ECE on target test
- Often drops fast then plateaus — interesting if it doesn't

### E3. Confidence histograms split by correct/incorrect
- Two overlaid histograms per iteration (correct in green, wrong in red)
- Good models have wide separation; bad models overlap

### E4. Entropy as proxy for OOD-ness
- Mean per-token predictive entropy on source vs. target test
- Should converge as target data is added

---

## F. Statistical reliability (must-have)

### F1. Variance bands on every iteration plot
- Always plot mean ± std (or 95% CI) across seeds. No single-seed numbers in the final report.

### F2. Paired significance test between iterations
- For "is iter_k significantly better than iter_(k−1) on target test?"
- **McNemar's test** on per-example binary correctness (paired across the same test sentences)
- Or bootstrap test on the F1 difference

### F3. Bootstrap CIs on F1
- Resample the test set with replacement 1000 times, compute F1 each time
- Report median + 2.5/97.5 percentiles — much more honest than a single point estimate

### F4. Effect-size table
- Cohen's d on F1 between adjacent iterations across seeds
- d > 0.8 = large effect, d ∈ [0.5, 0.8] = medium, etc.

### F5. Per-seed spaghetti plot
- One line per seed (don't average), faint, with mean overlaid bold
- Catches the case where one seed dominates the average

---

## G. Data-efficiency / sample-level analyses

### G1. Marginal value plot
- X: iteration
- Y: ΔF1 per added target example = (F1_k − F1_{k−1}) / n_added
- Diminishing returns is visible immediately; spikes suggest "this batch had exceptionally informative examples"

### G2. Cumulative learning curve in log-x
- Same as A1 but with log-scaled X-axis
- Often reveals the curve is roughly linear in log(n) — useful for extrapolating "how much more would we need?"

### G3. Power-law / saturation fit
- Fit F1(n) ≈ a − b·n^(−c) (or similar) and report fitted parameters + extrapolated asymptote
- The asymptote estimates "what's the best this approach could ever do, given more data?"

### G4. Active-learning–style retrospective
- If your friend can save which target examples got *correctly classified at iter_0* vs. those that didn't, you can ask: *"if we'd only added the hard ones, how much faster would we have converged?"*
- Plot: F1 vs. n with random selection (actual) vs. hypothetical hard-first selection

---

## H. Vocabulary / data-overlap analyses

### H1. Jaccard vocab overlap × F1 scatter
- One point per iteration: (Jaccard(train_mix_vocab, target_test_vocab), target_test_F1)
- Tests: is performance simply a function of vocabulary coverage?

### H2. OOV rate over iterations
- X: iteration, Y: fraction of target-test tokens unseen in current training mix
- Should drop monotonically; correlate with F1

### H3. Per-entity-type OOV rate
- For each type, what fraction of its surface forms in target test were unseen during training?
- Heatmap of (type × iteration)

---

## I. Training-dynamics analyses

### I1. Loss curves overlay
- One subplot per iteration; X = epoch, Y = train/dev loss
- Reveals if more target data changes the optimal stopping point

### I2. Best-epoch trend
- X: iteration, Y: best epoch number (when early stopping triggered)
- Often: more data → needs more epochs; or needs fewer if data is "easier"

### I3. Generalization gap
- (train_F1 − dev_F1) per iteration
- Shrinks as more in-domain data is added — quantifies overfitting reduction

---

## J. Cost / efficiency analyses

### J1. F1 vs. wall-clock time
- X: cumulative training time across iterations, Y: target_F1
- Compares "iterative retraining" to a hypothetical "train once on everything"

### J2. F1 per labeled example
- F1 gain divided by labeling cost (assuming roughly fixed cost per example)
- Useful for the "is more annotation worth it?" framing

### J3. Pareto front: source F1 vs. target F1
- Each iteration is a point in (source_F1, target_F1) space
- Connect them; the Pareto-optimal subset is your "deployment menu"

---

## K. Qualitative analyses (5–15 examples each)

### K1. Flipped-correct examples
Sentences the model got *wrong* at iter_0 and *right* at iter_N. Show tokens, gold spans, both predictions. These are the success stories.

### K2. Flipped-incorrect examples
Sentences the model got right at iter_0 and wrong at iter_N. Counterintuitive but real — these reveal interference.

### K3. Stubborn examples
Wrong at every iteration. Often reveal annotation issues or genuinely hard structure.

### K4. Confidence-flip examples
Same prediction across iterations, but confidence changed dramatically. Reveals what the model "learned" without changing output.

---

## L. Master summary tables

### L1. Headline results table
| Setting | Source F1 | Target F1 | Universal F1 | Mean ± std | n_params | Train min |
|---------|-----------|-----------|--------------|------------|----------|-----------|
| iter_0 (source only) | … | … | … | … | … | … |
| iter_N (full target) | … | … | … | … | … | … |
| target-only baseline | … | … | … | … | … | … |
| upper bound | … | … | … | … | … | … |

### L2. Per-type results at the chosen iteration
Per-type P/R/F1/support table at iter_N for source test and target test, side by side.

### L3. Significance summary
Pairwise McNemar p-values between key iterations (iter_0, iter_quarter, iter_half, iter_N) on target test.

---

## Recommended reporting order (for the writeup)

1. **Headline plot (A1)** + saturation table (A3) → answers RQ1
2. **Per-type heatmap (B1)** + flipped examples (K1, K2) → answers RQ2
3. **Forgetting curve (D1)** + BWT scalar (D2) + source-test confusion delta (C4) → answers RQ3
4. **Variance bands + significance (F1, F2)** → answers RQ4
5. **Calibration (E1, E2)** as a "bonus finding"
6. **Diminishing returns (G1, G3)** as a practical takeaway
7. **Qualitative examples (K1–K4)** sprinkled throughout

---

## Minimum viable analysis (if time is short)
If you only have time for 5 things:
1. Data-efficiency curve with seeds (A1) — *the one plot everyone will look at*
2. Source vs. target trade-off (A2) — shows you thought about forgetting
3. Per-type heatmap (B1) — shows granular understanding
4. Confusion delta matrix (C2) — shows what kind of errors got fixed
5. Significance test or bootstrap CI (F2/F3) — shows the result is real

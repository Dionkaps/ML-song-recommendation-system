# Explicit Decisions Report

Date: 2026-03-15

## Scope

This report documents the implementation of:

- `## Decisions To Make Explicitly`

from:

- `docs/reports/implementation_todo_20260314.md`

This was not just a prose cleanup pass. The goal was to convert the remaining
open decision items into:

- explicit repo-level policy
- synchronized code defaults
- documented product behavior
- reproducible evaluation evidence

## Final decisions

The following decisions are now explicit and implemented.

### 1. Cluster granularity

Decision:

- prefer broad macro-style clusters, not many micro-style clusters
- target occupied-cluster range for the supported GMM baseline: `4..8`
- current reference target: `4`

Why this was chosen:

- the supported product path needs stable, navigable recommendation neighborhoods
- the current selected production GMM already lands at `4` clusters
- `K-Means` at `2` clusters is operationally simple but likely too coarse to become
  the product's target granularity policy
- `HDBSCAN` does not provide a reliable multi-cluster macro-style partition under the
  current baseline because one giant cluster dominates and a noise partition remains

Implementation status:

- encoded in `config/feature_vars.py`
- surfaced in `config/experiment_profiles.py`
- documented in `docs/DECISION_POLICY.md`
- evaluated in `output/metrics/recommended_production_summary.json`

Current status:

- pass
- selected GMM cluster count: `4`
- target range: `4..8`

### 2. Minimum acceptable GMM stability threshold

Decision:

- a production-default GMM must clear all of the following:
  - subsample pairwise median ARI `>= 0.90`
  - subsample pairwise mean ARI `>= 0.75`
  - reference median ARI `>= 0.90`
  - per-cluster median best-match Jaccard `>= 0.90`

Why this was chosen:

- median-only stability can hide a weak resampling tail
- mean-only stability can over-penalize an otherwise usable model
- reference-vs-full alignment matters because the chosen model must be reproducible
  against the full saved baseline, not just internally consistent in pairwise terms
- per-cluster stability matters because a globally acceptable ARI can still hide a
  fragile subset of clusters

Implementation status:

- encoded in `config/feature_vars.py`
- surfaced in the canonical decision contract in `config/experiment_profiles.py`
- evaluated automatically in `scripts/analysis/evaluate_clustering.py`

Current status for the supported GMM baseline:

- pass
- subsample median ARI: `0.9058881806326302`
- subsample mean ARI: `0.8005308661704484`
- reference median ARI: `0.9357274570629188`
- per-cluster median Jaccard: `0.9509136227934182`

### 3. Uncertain GMM assignments

Decision:

- show uncertain GMM assignments normally by default
- do not hard-filter uncertain assignments by default
- keep posterior-weighted ranking and hard confidence/posterior thresholds as
  optional controls

Operational default:

- ranking method: `distance`
- minimum assignment confidence: `0.0`
- minimum selected-cluster posterior: `0.0`

Why this was chosen:

- hard filtering still looked too likely to hurt recommendation coverage
- posterior-weighted ranking sounded attractive in principle, but an explicit
  evaluation run showed that making it the default degraded GMM's proxy
  recommendation performance enough to worsen its overall comparison ranking
- keeping normal distance ranking as default preserves the best currently measured
  recommendation behavior, while still leaving the probabilistic controls available
  for manual/operator use

Implementation status:

- encoded in `config/feature_vars.py`
- surfaced in `config/experiment_profiles.py`
- wired into `src/ui/modern_ui.py`
- used as the default in `scripts/analysis/evaluate_clustering.py`
- documented in `docs/DECISION_POLICY.md`

Current status:

- aligned
- active evaluation default ranking method: `distance`
- active default thresholds: `0.0` / `0.0`

### 4. MSD numeric metadata return gate

Decision:

- MSD numeric metadata returns only as an explicit experiment
- it must not silently re-enter the supported default path

Required gate:

- live audio-backed numeric MSD coverage `>= 98%`
- missing live audio-backed rows `<= 100`
- clean metadata/schema audit
- no silent fallback or silent imputation path
- fresh explicit comparison rerun under the experiment profile

Why this was chosen:

- the supported baseline is audio-only and fully populated today
- partial MSD coverage would reintroduce either silent dropping or silent fallback
  behavior, both of which would make results harder to interpret
- the current metadata state is still too incomplete for safe default use

Implementation status:

- encoded in `config/feature_vars.py`
- surfaced in `config/experiment_profiles.py`
- documented in `docs/DECISION_POLICY.md`
- automatically checked in `scripts/analysis/evaluate_clustering.py`

Current status:

- not ready
- current live audio-backed rows: `5535`
- rows with numeric MSD features: `4783`
- coverage: `0.8641373080397471` (`86.41%`)
- missing audio-backed rows: `752`
- saved source: `data/songs_schema_summary.json`

## Files changed

### New files

- `docs/DECISION_POLICY.md`
- `docs/reports/explicit_decisions_report_20260315.md`

### Updated files

- `config/feature_vars.py`
- `config/experiment_profiles.py`
- `src/ui/modern_ui.py`
- `scripts/analysis/evaluate_clustering.py`
- `docs/RECOMMENDED_PRODUCTION_BASELINE.md`
- `docs/SUPPORTED_BASELINE.md`
- `README.md`
- `docs/README.md`
- `docs/reports/implementation_todo_20260314.md`

## Main implementation work

### 1. Repo-level decision policy

The open decisions are no longer hidden in a TODO section.

They are now encoded in `config/feature_vars.py` as explicit baseline policy
constants:

- cluster granularity policy and target range
- GMM stability thresholds
- uncertainty-handling default behavior
- MSD restoration gate

This makes the operational defaults inspectable from code rather than requiring
someone to read historical reports.

### 2. Canonical contract export

`config/experiment_profiles.py` now exposes the explicit decision policy through
the canonical baseline contract.

That means:

- experiment manifests can carry the same decision context as the code defaults
- baseline docs can point to one machine-readable contract
- future work no longer has to infer policy from scattered report text

The canonical contract now includes:

- `decision_policy.cluster_granularity`
- `decision_policy.gmm_stability_gate`
- `decision_policy.uncertain_assignments`
- `decision_policy.msd_restore_gate`

### 3. UI/evaluation synchronization

`src/ui/modern_ui.py` and `scripts/analysis/evaluate_clustering.py` now read the
same uncertainty-handling defaults from config.

This matters because the repo had already added:

- assignment confidence
- posterior probabilities
- optional posterior-weighted ranking
- optional confidence/posterior filters

but the product had not decided what the default behavior should be.

After this work:

- the UI default matches the explicit decision policy
- the evaluation default matches the same policy
- decision reports can check whether the current run is aligned or running an override

### 4. Automated decision assessment

`scripts/analysis/evaluate_clustering.py` now writes both:

- `decision_policy`
- `decision_assessment`

into the generated evaluation summary JSON.

That assessment currently checks:

- whether GMM cluster count matches the target granularity policy
- whether the current GMM clears the explicit stability gate
- whether the active evaluation run is aligned with the uncertainty-handling default
- whether the MSD restoration gate is currently met

This turns the decisions into something the repo can evaluate, not just describe.

## Evaluation evidence used for the uncertainty-handling decision

This was the one decision that required a direct side-by-side runtime comparison.

Two full evaluation passes were run against the same current production artifacts.

### A. Final chosen policy run: distance default

Command:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/evaluate_clustering.py `
  --methods kmeans gmm hdbscan `
  --artifact-dir output/clustering_results `
  --metrics-dir output/metrics `
  --prefix recommended_production `
  --subsample-runs 50 `
  --subsample-fraction 0.8 `
  --seed-runs 10 `
  --stability-jobs 1
```

Saved outputs:

- `output/metrics/recommended_production_summary.json`
- `output/metrics/recommended_production_comparison.csv`
- `output/metrics/recommended_production_comparison_report.md`

Key result for `GMM` at `K=10`:

- genre precision@10: `0.02693766937669377`
- genre hit rate@10: `0.201806684733514`
- artist precision@10: `0.004101174345076785`
- artist hit rate@10: `0.03739837398373984`
- catalog coverage: `0.9878952122854562`
- overall rank: `2`

### B. Explicit alternative run: posterior-weighted override

Command:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/evaluate_clustering.py `
  --methods kmeans gmm hdbscan `
  --artifact-dir output/clustering_results `
  --metrics-dir output/metrics `
  --prefix recommended_production_posterior_weighted `
  --ranking-method posterior_weighted `
  --subsample-runs 50 `
  --subsample-fraction 0.8 `
  --seed-runs 10 `
  --stability-jobs 1
```

Saved outputs:

- `output/metrics/recommended_production_posterior_weighted_summary.json`
- `output/metrics/recommended_production_posterior_weighted_comparison.csv`
- `output/metrics/recommended_production_posterior_weighted_comparison_report.md`

Key result for `GMM` at `K=10`:

- genre precision@10: `0.026160794941282747`
- genre hit rate@10: `0.198193315266486`
- artist precision@10: `0.004083107497741645`
- artist hit rate@10: `0.03685636856368564`
- catalog coverage: `0.9248419150858175`
- overall rank: `3`

### Measured effect of the posterior-weighted override on GMM

Difference versus the final chosen distance default:

- genre precision@10: down by about `0.000776874435411023`
- genre hit rate@10: down by about `0.003613369467028`
- artist precision@10: down by about `0.00001806684733514`
- artist hit rate@10: down by about `0.0005420054200542`
- catalog coverage: down by about `0.0630532971996387`
- overall rank: dropped from `#2` to `#3`

Interpretation:

- posterior-weighting did not improve the recommendation-first proxy picture
- the largest practical regression was catalog coverage
- the smaller metric deltas would have been tolerable on their own, but the
  coverage drop plus the rank drop made the override a poor choice for the
  supported product default

This is why the final explicit decision is:

- keep distance ranking as the default
- keep posterior-weighting as an optional operator mode

## Final measured decision status

The final chosen default policy, as saved in:

- `output/metrics/recommended_production_summary.json`

reports:

- `cluster_granularity.status = pass`
- `gmm_stability_gate.status = pass`
- `uncertain_assignments.status = aligned`
- `msd_restore_gate.status = not_ready`

That is the cleanest possible end-state for this pass:

- the decisions are explicit
- the chosen defaults are reflected in code
- the current baseline actually passes the decisions that should be pass/fail
- the MSD return decision remains intentionally blocked because the numbers do not clear the gate

## Verification

### Static verification

The following command succeeded:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  config/feature_vars.py `
  config/experiment_profiles.py `
  src/ui/modern_ui.py `
  scripts/analysis/evaluate_clustering.py
```

### Runtime verification

Executed successfully:

- one full production evaluation under the final chosen default policy
- one full production evaluation under the explicit posterior-weighted override

Observed wall times:

- distance-default evaluation: about `163.8` seconds
- posterior-weighted comparison evaluation: about `166.2` seconds

## Important non-changes

This pass did not:

- change the supported clustering method away from `GMM`
- rerun the full clustering training/search stack
- re-enable MSD numeric metadata
- hard-enable confidence/posterior filtering in the UI

Those were intentional non-changes.

The purpose here was to make the remaining open policy decisions explicit and
evidence-based without introducing unrelated baseline churn.

## Conclusion

`## Decisions To Make Explicitly` is now implemented.

The repo now has:

- a durable decision-policy doc
- machine-readable decision policy in the canonical baseline contract
- synchronized UI/evaluation defaults
- automatic pass/fail assessment for the active policy
- saved evidence showing why the uncertainty-handling default remains normal
  distance ranking instead of posterior-weighted ranking

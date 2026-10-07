# Model Validation Study — Assessment 2
## FMCG Retail Store Segmentation

### Overview

A controlled improvement study was conducted to determine whether the baseline
K-Means k=3 segmentation could be genuinely improved through better feature
representation or clustering methodology.

**Experiments performed:** 68
**Feature variants:** 4 (Baseline, Ratio-normalised, Hybrid behavioral, RobustScaler)
**Algorithms:** K-Means, Ward, GMM
**k range:** 2-8

### Baseline (Frozen Production Model)

| Metric | Value |
|---|---|
| Algorithm | K-Means |
| k | 3 |
| Stores | 761 |
| Silhouette | 0.4857 |
| Davies-Bouldin | 0.6703 |
| Calinski-Harabasz | 434.2773 |
| Mean ARI | 0.7853 |
| Outliers | 38 |
| Cluster Distribution | 10 / 424 / 327 |

### Feature Variants Tested

**A — Baseline:** Log1p + StandardScaler on original feature set.
Features kept after correlation pruning (>0.95): SPEND_sum_sum, SPEND_sum_mean,
SPEND_sum_std, BASKET_nunique_sum, WEEK_nunique_sum, SPEND_cv

**B — Ratio-normalised:** Replaced absolute volume metrics with per-week and per-transaction
ratios (spend_per_week, qty_per_txn, spend_per_txn, basket_rate, prod_per_txn, spend_cv,
active_weeks). Rationale: test whether absolute scale dominates clustering.

**C — Hybrid behavioral:** Combination of log-normalised rate features and volatility metrics
(spend_per_week, spend_cv, spend_std, qty_per_txn, prod_per_txn, active_weeks).

**D — RobustScaler:** Same baseline features with RobustScaler instead of StandardScaler.
Rationale: test whether outlier sensitivity in StandardScaler drives the 10-store cluster.

### Key Finding: The 10-Store Cluster

The most important question tested was whether the 10-store Cluster 0 is a
genuine structural group or an artefact of scale-dominated preprocessing.

Evidence from the experiment:

| Feature | 10-Store Mean | Overall Mean | Z-Score |
|---|---|---|---|
| SPEND_sum_sum | 81,506 | 78,428 | +0.06 (near average) |
| SPEND_sum_mean | 874 | 674 | +0.46 |
| SPEND_sum_std | 604 | 154 | +3.39 |
| BASKET_nunique_sum | 5,092 | 6,224 | -0.40 |
| WEEK_nunique_sum | 87 | 116 | -5.77 |
| SPEND_cv | 0.615 | 0.262 | +2.80 |

Conclusion: These stores are NOT the largest stores by revenue. They are
stores with dramatically fewer active weeks (87 vs 116 average; z=-5.77) and
extreme spend volatility (Spend CV z=+2.80). They are intermittently operating
stores with near-average total revenue but abnormal operational patterns.

These stores exhibit intermittent activity and unusually high volatility and may
warrant investigation into operational or seasonal factors.

The cluster persisted across ALL four feature variants:
- B_Ratio k=3: sizes [10, 282, 469]
- C_Hybrid k=3: sizes [10, 268, 483]
- D_Robust k=2: sizes [10, 751] (confirming they are genuine structural outliers)

Baseline k=3 ARI across seeds: [0.9897, 1.0, 0.9897, 1.0, 1.0] — Mean: 0.9959
(This is supporting evidence from the controlled experiment, not the official
production ARI. The official final ARI remains 0.7853 from the production pipeline.)

### Why Alternatives Were Rejected

| Alternative | Reason Rejected |
|---|---|
| K-Means/Ward k=2 | Collapses the 10-store intermittent group; loses business nuance; only high vs low volume |
| K-Means k=4+ | Produces clusters of 3 stores; too small for any commercial action |
| Ratio features (B) | Lower Silhouette, higher DB, lower CH vs baseline; 10-store cluster persists anyway |
| Hybrid features (C) | Same: weaker metrics, same cluster structure |
| RobustScaler k=3 | Produces 750/5/6 split — severely imbalanced, not usable |
| Best alternative (Ward k=2) | Silhouette 0.4536 vs 0.4857 baseline; loses the 10-store segment |

### Recommendation

**KEEP BASELINE — K-Means k=3**

No experimental candidate improved on all primary criteria simultaneously.
The 10-store intermittent/high-volatility segment is structurally valid and
its identification is a genuine additional insight from the segmentation.

### Files

- `outputs/experiment_results.csv` — All 68 experiment results
- `outputs/shortlisted_candidates.csv` — Top 10 candidates after filtering
- `outputs/small_cluster_profile.csv` — Feature profile of the 10-store cluster
- `run_experiment.py` — Reproducible experiment script

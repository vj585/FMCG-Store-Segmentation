# Assessment 2 Requirements Audit

## Overall Result

PASS

## Requirement Checklist

| # | Requirement | Status | Evidence | File |
|---|---|---|---|---|
| 1 | Extract requirements strictly for Assessment 2. | PASS | Completed manually from the prompt context. | Prompt constraints |
| 2 | Use aggregated store-level data. | PASS | Loads `store_features_base.parquet` exclusively. | `src/a2_segmentation.py` |
| 3 | Apply Log1p and standard scaling to features. | PASS | `np.log1p` and `StandardScaler` implemented. | `src/a2_segmentation.py` |
| 4 | Evaluate candidate algorithms and multiple k values. | PASS | Iterates k=2..10 for both K-Means and Agglomerative. | `reports/kmeans_metrics.csv` |
| 5 | Measure stability (ARI) and evaluate sizes. | PASS | Calculates size balancing and 5-seed ARI. | `reports/cluster_stability.csv` |
| 6 | Select justified K balancing stability/metrics. | PASS | Validated k=3 selection natively documented over k=2. | `reports/FINAL_REPORT.md` |
| 7 | Profile segments by actual characteristics. | PASS | Dynamic Z-score mapping used to generate profiles. | `outputs/cluster_profiles.csv` |
| 8 | Assign actionable business recommendations. | PASS | Strategies mapped securely based on metric outputs. | `outputs/segment_recommendations.csv` |
| 9 | Perform structural outlier analysis. | PASS | 38 outliers identified via Euclidean centroid distance. | `outputs/cluster_outliers.csv` |
| 10 | Create professional Dash frontend avoiding raw data. | PASS | Multi-page Dash layout natively consumes CSV outputs. | `app.py` |
| 11 | Ensure reproducibility (no absolute paths/secrets). | PASS | Relative paths and `requirements.txt` enforced. | `run_pipeline.py` / `app.py` |
| 12 | Ensure GitHub readiness (.gitignore raw files). | PASS | Local repo initialized, raw data heavily ignored. | `.gitignore` |

## 1. Dataset & Data Preparation
- **Status:** PASS
- Aggregation is strictly enforced at the store-level, protecting the local application from computationally blowing up trying to handle raw transactions. Administrative IDs are stripped securely before evaluation.

## 2. Feature Engineering
- **Status:** PASS
- Heavy skewing properly balanced using `log1p`. Variables correctly bounded by `StandardScaler` to prevent gross volume metrics from destroying granular behavioral analysis.

## 3. Clustering Methodology
- **Status:** PASS
- Successfully iterated over traditional boundaries, natively rejecting an imbalanced `k=2` solution in favor of a mathematically stable (`Mean ARI = 0.7853`) and structurally balanced (`Largest cluster = 55.72%`) `k=3` approach.

## 4. Cluster Profiling
- **Status:** PASS
- Segments natively profiled according to strict centroid offsets (z-scores relative to standard deviations), producing accurate "High Volume" vs "Balanced" descriptions automatically.

## 5. Business Recommendations
- **Status:** PASS
- Marketing, Inventory, and Strategic actions mapped directly to mathematical clusters without leveraging fabricated causal assumptions.

## 6. Outlier Analysis
- **Status:** PASS
- Top 5% anomaly mapping correctly isolated 38 specific structural outliers into a dedicated table without classifying them arbitrarily as errors.

## 7. Frontend
- **Status:** PASS
- Plotly Dash architecture cleanly executed using responsive cards and data tables. Natively separates machine-learning pipeline processing from the UI loading phase.

## 8. Report
- **Status:** PASS
- `FINAL_REPORT.md` generated with zero missing sections, appropriately quoting stability bounds and rejecting the false-positive `k=2` signal explicitly.

## 9. Reproducibility
- **Status:** PASS
- Scripts executed cleanly multiple times on this instance. `requirements.txt` formally bound to the repository.

## 10. GitHub Readiness
- **Status:** PASS
- Commits generated natively, raw transactional folders specifically blacklisted from `.gitignore`, avoiding dangerous `git push` bottlenecks.

## Missing / Partial Requirements
None.

## Recommended Fixes
None. Project is fully validated against prompt definitions.

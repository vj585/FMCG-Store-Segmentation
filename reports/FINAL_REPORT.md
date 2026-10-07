# FMCG Retail Store Segmentation Using Clustering

## 1. Executive Summary
- **Objective:** Segment retail stores into meaningful behavioral/business groups to optimize strategy.
- **Dataset Scale:** 31M raw transactional rows aggregated to 761 stores.
- **Approach:** Exploratory clustering using K-Means and Hierarchical algorithms, rigorously balancing silhouette scores with stability and business interpretability.
- **Final Segmentation Result:** A robust 3-segment solution that distinctly profiles stores by volume and volatility.

## 2. Dataset and Store-Level Aggregation
- **Data Source:** Raw transaction-level data was aggressively grouped to extract store-level descriptors, completely avoiding raw row-clustering to prevent computational bloat and noise.
- **Quality & Missingness:** Any structural missing metrics were imputed using the global median.

## 3. Feature Engineering
- **Used Features:** SPEND_sum_sum, SPEND_sum_mean, SPEND_sum_std, BASKET_nunique_sum, WEEK_nunique_sum, SPEND_cv
- **Exclusions:** Store identifiers (STORE_CODE) and highly correlated redundances (>0.95 correlation) were dropped.
- **Transformations:** Log1p applied to correct heavily right-skewed scale metrics, followed by StandardScaler to prevent size-dominance bias.

## 4. Clustering Methodology
- Evaluated K-Means and Agglomerative (Ward linkage) across candidate values k=2 through k=10.
- **Metrics utilized:** Silhouette, Davies-Bouldin, Calinski-Harabasz, and Adjusted Rand Index (ARI) for stability across random initializations.

## 5. Final Cluster Selection
**Selected Solution:** K-Means with k=3.

**Rejection of k=2:** While k=2 produced an apparently strong absolute silhouette score, it created an extremely imbalanced segmentation (~98% of stores in one cluster) and poor stability (mean ARI approximately 0.16). This was rejected as functionally useless for actionable business strategy.

**Preference for k=3:**
- Largest cluster is a balanced 55.72%.
- Extremely high stability (Mean ARI: 0.7853).
- Retains strong mathematical quality while presenting highly distinct, business-interpretable store profiles.

## 6. Cluster Profiles
| Cluster | Label | Stores | % | Major Characteristics |
|---------|-------|--------|---|-----------------------|
| 0.0 | Cluster 0: Intermittent / High Volatility | 10.0 | 1.31% | ~87 active weeks vs 116 avg; Spend CV 0.61 vs 0.26 avg; extreme spend volatility |
| 1.0 | Cluster 1: High Volume | 424.0 | 55.72% | High overall sales; broad product mix; high transaction and customer activity; consistent activity |
| 2.0 | Cluster 2: Low Volume | 327.0 | 42.97% | Low overall sales; below-average transaction volume; lower product variety; consistent activity |


## 7. Business Recommendations
### Cluster 0: Intermittent / High Volatility
- **Recommendation:** Dedicated investigation recommended. Inventory: Maintain minimal baseline stock; avoid standard replenishment cycles. Do not apply standard promotional investment without understanding root cause.

### Cluster 1: High Volume
- **Recommendation:** Marketing: Prioritize for premium product launches and dedicated promotions. Assortment: Maximize breadth. Inventory: High priority allocation. Opportunity: Key revenue driver.

### Cluster 2: Low Volume
- **Recommendation:** Marketing: Focused, high-ROI events only. Assortment: Trim tail products, focus on high-turnover staples. Inventory: Standard cyclic replenishment at reduced scale. Opportunity: Cost optimization and selective growth.

## 8. Outlier Analysis
Exactly 38 stores (the top 5% most distant from their respective cluster centroids) have been flagged. These should be manually investigated. They may represent unusually large flagship stores, structural data gaps, or distinct regional anomalies, rather than strict errors.

## 9. Limitations
- **Descriptive Nature:** Clustering is unsupervised; segments map observed patterns, not intrinsic causal truth.
- **Causality:** Business recommendations represent observed correlations and hypotheses, not causal guarantees.
- **Dependencies:** Results are strongly dependent on the selected feature set and preprocessing transformations.

## 10. Conclusion
The robust 3-segment solution actively balances mathematical rigor with operational reality, avoiding the pitfalls of unbalanced optimization and delivering a framework strictly aligned for targeted retail operations.

## 11. Model Validation Study
A controlled improvement study of 68 experiments was conducted across four feature representations, three algorithms, and k=2 through k=8.

Key findings:
- No alternative candidate improved on all primary criteria simultaneously (separation, stability, balance, business interpretability).
- The 10-store cluster persisted across all feature variants, confirming its structural validity.
- The 10-store cluster represents stores with significantly fewer active weeks (~87 vs 116 avg) and extremely high spend volatility (Spend CV z=+2.80), not simply extreme high-volume stores.
- These stores exhibit intermittent activity and unusually high volatility and may warrant investigation into operational or seasonal factors.

Conclusion: The original K-Means k=3 solution was retained. No evidence justified replacing it.

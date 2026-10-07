# Assessment 2 Phase 3 Report: Retail Store Segmentation

## 1. Business Objective
Segment retail stores into meaningful behavioral/business groups using exploratory clustering.

## 2. Dataset & Grain
Store-level grain. Total stores: 761

## 3. Feature Processing
Features were imputed with median (if missing), log1p transformed (for heavily skewed scale metrics), and standardized using StandardScaler to prevent scale dominance.

## 4. Feature Selection
Features with > 0.95 correlation were pruned to reduce multicollinearity. Features retained: 6

## 5. Model Evaluation
K-Means and Hierarchical (Ward) were evaluated from k=2 to k=10 using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.

## 6. Cluster Count Selection
Silhouette score identified k=2 as the strongest candidate on separation alone, with a score of 0.7213. However, k=2 produced a severely imbalanced segmentation (751 vs 10 stores) and poor stability (mean ARI = 0.1620). Therefore, k=2 was rejected.

The final k=3 solution was selected because it provided a better overall balance of cluster stability, cluster-size balance, separation, and business interpretability. The final k=3 solution achieved a silhouette score of 0.4857 and mean ARI of 0.7853.

## 7. Stability Analysis
Stability was evaluated using Adjusted Rand Index (ARI) across multiple random seeds.

The rejected k=2 solution had a mean ARI of 0.1620, indicating poor consistency of cluster assignments across runs.

The final k=3 solution achieved a mean ARI of 0.7853, indicating high consistency of assignments and supporting the stability of the selected segmentation.

## 8. Outlier Analysis
Stores in the top 5% (95th percentile) of Euclidean distance to their assigned cluster centroid were flagged as outliers for business review. A total of 38 stores were identified. These stores represent atypical store profiles relative to their assigned cluster and are not automatically poor-performing stores.

## 9. Business Interpretation
See `segment_recommendations.csv`. Labels were assigned strictly empirically using z-scores relative to global means.

## 10. Limitations
Clustering is unsupervised exploratory analysis. It groups stores by observed characteristics but does not guarantee causation or intrinsic "types".

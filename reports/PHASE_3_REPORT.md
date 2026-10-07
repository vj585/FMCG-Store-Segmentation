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
k=2 was selected algorithmically by maximizing the Silhouette score.

## 7. Stability Analysis
Mean ARI across random seeds: 0.1620. Indicates consistency of assignments.

## 8. Outlier Analysis
Stores in the top 95th percentile of Euclidean distance to their assigned cluster centroid were flagged as outliers for business review (e.g., massive outliers in scale).

## 9. Business Interpretation
See `segment_recommendations.csv`. Labels were assigned strictly empirically using z-scores relative to global means.

## 10. Limitations
Clustering is unsupervised exploratory analysis. It groups stores by observed characteristics but does not guarantee causation or intrinsic "types".

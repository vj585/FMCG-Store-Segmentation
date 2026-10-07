# FMCG Retail Store Segmentation

## Project Objective
Segment retail stores into meaningful behavioral and business groups using clustering algorithms to optimize assortment, pricing, and promotional strategies.

## Dataset Description
The analysis leverages a processed store-level dataset aggregated from 31 million raw transaction rows. It features robust descriptors including total spend volume, unique transaction counts, unique products (category mix), and spend volatility (coefficient of variation).

## Methodology
- **Data Preparation**: Handled missing values using median imputation. Extracted identifiers.
- **Feature Engineering**: Standardized variables to prevent scale dominance and applied `log1p` to severely right-skewed sales distribution features. Highly correlated components were explicitly flagged and dropped.
- **Algorithms**: Evaluated K-Means and Agglomerative Hierarchical Clustering (Ward linkage) across k=2 to k=10.
- **Cluster Count Selection**: Determined robustly via mathematical evaluation of Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
- **Evaluation**: Assessed cluster stability across multiple random seeds using the Adjusted Rand Index (ARI). Outliers identified using centroid distances.

## How to Install Dependencies
This project uses standard data science libraries in Python 3:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## How to Run
### 1. Run the Data Pipeline
To completely reproduce the Phase 3 segmentation modeling, clustering, and reporting:
```bash
cd Assessment-2-Store-Segmentation
python run_pipeline.py
```
This generates all clustering outputs, models, and metric evaluations into the `outputs/` and `reports/` directories.

### 2. Run the Dashboard
To start the Interactive Plotly Dash Frontend:
```bash
cd Assessment-2-Store-Segmentation
python app.py
```
Then open `http://127.0.0.1:8050/` in your browser. The dashboard natively reads the offline-generated CSVs to ensure instant load times without aggressively processing raw transactions on launch.

## Dashboard Pages
1. **Overview**: Key KPI metrics, high-level cluster sizing, and standardized profile heatmap.
2. **Segments**: Deep dives into the three specific segments and tailored business actions.
3. **Store Explorer**: Interactive filterable/sortable table encompassing the 761 stores.
4. **Outliers**: View the 38 structural outliers mathematically distant from central network behavior.
5. **Methodology**: Explanation of k-selection, diagnostic metrics comparison, and final ARI stability validation.
- **Plots**: `outputs/plots/` (Heatmaps, dendrograms, validation elbows, PCA)
- **Reports**: `reports/FINAL_REPORT.md` (Comprehensive documentation)

## Key Results
- **k=2** generated the highest Silhouette score. The network strictly partitions between a massive stable population and a tiny cohort of distinct high-volume/volatile stores.
- Outlier detection isolated exactly 5% of structurally atypical store profiles for manual review.

## Limitations
Clustering is fundamentally exploratory and unsupervised. Segments indicate mathematical similarities across observed historical characteristics; they do not dictate absolute distinct groups nor can they assert causation in revenue strategies without A/B testing.

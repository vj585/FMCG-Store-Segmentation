# EDA & Data Quality Report — Assessment 2

## 1. Data Source & Volume
- **Input:** 117 raw transaction CSV files (`transactions_*.csv`).
- **Raw Volume:** Approximately 31 million transaction rows.
- **Aggregation:** Data was strictly aggregated to the store level (grouping by `STORE_CODE`) to extract behavioral features. Raw row-level clustering was intentionally avoided to prevent computational bloat and noise.

## 2. Store Coverage & Temporal Scope
- **Total Stores:** 761 unique retail stores extracted from the raw data.
- **Temporal Coverage:** The aggregated features measure behavior across an observed timeline (represented by `WEEK_nunique_sum`). Most stores exhibit ~116-117 active weeks, though a small subset displays significantly intermittent activity (~54-87 weeks).

## 3. Data Quality & Missingness
- **Missing Values:** No null values exist in the primary numeric aggregations (Sales, Transactions, Weeks).
- **Imputation:** Any structural missing metrics that emerge during aggregation (e.g., CV calculations yielding NaNs on zero-variance stores) were imputed using the global median for that feature to prevent dropping valid store entities.
- **Important Missing Fields:** The available analytical dataset contains POS transaction fields (product, store, quantity, spend, date). It **does not** contain pricing history, promotion flags, inventory levels, stockout records, or direct category hierarchies (premium vs value).

## 4. Feature Engineering & Distributions
Ten continuous behavioral features were extracted:
- **Sales Scale:** Total Spend, Average Weekly Spend, Total Quantity.
- **Customer & Basket:** Transaction Volume, Unique Customers, Average Basket Size.
- **Product Mix:** Product Variety (Unique Products).
- **Volatility & Coverage:** Spend Volatility (Std), Spend Variability (CV), Active Weeks.

**Skewness:** Features related to total volume (Spend, Quantity, Transactions) are heavily right-skewed (skewness ~ 0.35 to 0.61), indicating a long tail of very high-volume stores. To address this, a `Log1p` transformation is applied prior to clustering, followed by `StandardScaler`.

## 5. Feature Correlation & Redundancy
Highly correlated metrics (>0.95 Pearson correlation) were systematically identified to prevent volume-dominance bias in the clustering distance metric.
- **Redundant/Dropped:** Transaction Volume, Total Quantity, Unique Customers, and Product Variety were heavily collinear with Total Spend and each other.
- **Final Independent Feature Set Retained:** Total Spend (`SPEND_sum_sum`), Average Spend (`SPEND_sum_mean`), Spend Volatility (`SPEND_sum_std`), Average Basket Size (`BASKET_nunique_sum`), Active Weeks (`WEEK_nunique_sum`), and Spend Variability (`SPEND_cv`).

## 6. Limitations
- **Descriptive Only:** The extracted features represent observed historical correlations. They do not prove causal relationships.
- **Data Coverage:** Recommendations regarding inventory or promotions are hypotheses based on volatility and scale. Actual inventory optimization requires supply-chain data unavailable in this dataset.

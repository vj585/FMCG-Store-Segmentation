# -*- coding: utf-8 -*-
"""
Assessment 2 — Model Improvement Experiment
============================================
EXPERIMENTAL ONLY. Does NOT modify baseline outputs.
All results written to experiments/outputs/ only.

Baseline (frozen):
  K-Means k=3, Silhouette=0.4857, DB=0.6703, CH=434.2773,
  ARI=0.7853, Outliers=38, Clusters=[10, 424, 327]
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import skew

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(PROJ_ROOT, "data", "processed")
EXP_OUT    = os.path.join(PROJ_ROOT, "experiments", "outputs")
os.makedirs(EXP_OUT, exist_ok=True)

SEEDS = [10, 20, 30, 40, 50]

# ── Baseline definition (never mutated) ────────────────────────────────────
BASELINE = {
    "Algorithm": "K-Means", "k": 3,
    "Silhouette": 0.4857, "DB": 0.6703, "CH": 434.2773,
    "Mean_ARI": 0.7853, "Outliers": 38,
    "Clusters": "10 / 424 / 327", "Feature_Variant": "Baseline"
}

# ── Helpers ─────────────────────────────────────────────────────────────────
def ari_stability(X, labels, k, algo="kmeans"):
    scores = []
    for seed in SEEDS:
        if algo == "kmeans":
            test = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        elif algo == "ward":
            test = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        else:
            gm = GaussianMixture(n_components=k, random_state=seed, n_init=3)
            test = gm.fit_predict(X)
        scores.append(adjusted_rand_score(labels, test))
    return float(np.mean(scores)), float(np.min(scores))


def evaluate(X, labels, k, algo, variant, preproc):
    counts = pd.Series(labels).value_counts().sort_index()
    min_sz  = int(counts.min())
    max_sz  = int(counts.max())
    pct_max = round(max_sz / len(labels) * 100, 2)
    sil  = round(silhouette_score(X, labels), 4)
    db   = round(davies_bouldin_score(X, labels), 4)
    ch   = round(calinski_harabasz_score(X, labels), 4)
    mean_ari, min_ari = ari_stability(X, labels, k, algo)
    dist_str = " / ".join(str(c) for c in counts.values)
    interp   = "Imbalanced" if pct_max > 85 else "Acceptable"
    return {
        "Feature_Variant": variant,
        "Preprocessing": preproc,
        "Algorithm": algo,
        "k": k,
        "Silhouette": sil,
        "DB": db,
        "CH": ch,
        "Mean_ARI": round(mean_ari, 4),
        "Min_ARI": round(min_ari, 4),
        "Min_Cluster": min_sz,
        "Max_Cluster": max_sz,
        "Largest_%": pct_max,
        "Cluster_Distribution": dist_str,
        "Interpretability": interp,
    }


def run_kmeans_and_ward(X, k_range, variant, preproc, results):
    for k in k_range:
        km_labels  = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        results.append(evaluate(X, km_labels, k, "kmeans", variant, preproc))

        hc_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        results.append(evaluate(X, hc_labels, k, "ward", variant, preproc))


def gmm_experiments(X, k_range, variant, preproc, results):
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gm_labels = gm.fit_predict(X)
        counts = pd.Series(gm_labels).value_counts()
        if counts.min() >= 2:
            results.append(evaluate(X, gm_labels, k, "gmm", variant, preproc))


# ── Load raw features ──────────────────────────────────────────────────────
print("Loading processed data…")
df = pd.read_parquet(os.path.join(DATA_PROC, "store_features_base.parquet"))
print(f"  Stores: {len(df)}")

# Median-impute any missing
num_cols = [c for c in df.columns if c != "STORE_CODE" and pd.api.types.is_numeric_dtype(df[c])]
for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)


# ── FEATURE VARIANTS ───────────────────────────────────────────────────────

# A. Baseline features (replicate exactly as produced by a2_segmentation.py)
#    Log1p on volume cols → StandardScaler → drop corr>0.95
BASELINE_VOL_COLS = ["SPEND_sum_sum", "QUANTITY_sum_sum", "TXN_count_sum",
                     "BASKET_nunique_sum", "CUST_nunique_sum", "PROD_nunique_sum"]

def make_baseline_X(df):
    d = df.copy()
    for c in BASELINE_VOL_COLS:
        if c in d.columns:
            d[c] = np.log1p(d[c])
    feat_cols = [c for c in d.columns if c != "STORE_CODE"]
    scaled = pd.DataFrame(StandardScaler().fit_transform(d[feat_cols]), columns=feat_cols)
    corr = scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    final = [f for f in feat_cols if f not in to_drop]
    print(f"  [Baseline] features kept: {final}")
    return scaled[final].values, final


# B. Scale-controlled / ratio feature variant
#    Replace raw absolute volumes with per-week and per-transaction ratios.
#    Rationale: SPEND_sum_sum and SPEND_sum_mean both represent absolute
#    scale; replacing with rates removes size-dominance bias.
def make_ratio_X(df):
    d = df.copy()
    weeks = d["WEEK_nunique_sum"].clip(lower=1)
    txns  = d["TXN_count_sum"].clip(lower=1)

    d["spend_per_week"]    = d["SPEND_sum_sum"] / weeks          # avg weekly revenue
    d["qty_per_txn"]       = d["QUANTITY_sum_sum"] / txns        # basket size (units)
    d["spend_per_txn"]     = d["SPEND_sum_sum"] / txns           # avg transaction value
    d["basket_rate"]       = d["BASKET_nunique_sum"] / weeks     # unique baskets per week
    d["cust_per_week"]     = d["CUST_nunique_sum"] / weeks       # unique customers per week
    d["prod_per_txn"]      = d["PROD_nunique_sum"] / txns        # product diversity per txn
    d["spend_cv"]          = d["SPEND_cv"]                       # keep volatility
    d["active_weeks"]      = d["WEEK_nunique_sum"]               # keep activity level

    ratio_cols = ["spend_per_week", "qty_per_txn", "spend_per_txn",
                  "basket_rate", "cust_per_week", "prod_per_txn",
                  "spend_cv", "active_weeks"]
    sub = d[ratio_cols]
    # log1p on skewed ratio cols
    for c in ["spend_per_week", "spend_per_txn", "basket_rate", "cust_per_week", "prod_per_txn"]:
        sub = sub.copy()
        sub[c] = np.log1p(sub[c])
    scaled = pd.DataFrame(StandardScaler().fit_transform(sub), columns=ratio_cols)
    corr = scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    final = [f for f in ratio_cols if f not in to_drop]
    print(f"  [Ratio] features kept: {final}")
    return scaled[final].values, final


# C. Hybrid — keep spend volatility + ratio-normalized behavioral features
#    but also retain log-spend_per_week (behavioral scale), WEEK_nunique (activity).
#    Drop raw absolute scale features entirely.
def make_hybrid_X(df):
    d = df.copy()
    weeks = d["WEEK_nunique_sum"].clip(lower=1)
    txns  = d["TXN_count_sum"].clip(lower=1)

    d["spend_per_week"]  = np.log1p(d["SPEND_sum_sum"] / weeks)
    d["spend_cv"]        = d["SPEND_cv"]
    d["spend_std"]       = np.log1p(d["SPEND_sum_std"])
    d["qty_per_txn"]     = d["QUANTITY_sum_sum"] / txns
    d["cust_per_week"]   = np.log1p(d["CUST_nunique_sum"] / weeks)
    d["prod_per_txn"]    = np.log1p(d["PROD_nunique_sum"] / txns)
    d["active_weeks"]    = d["WEEK_nunique_sum"]

    hybrid_cols = ["spend_per_week", "spend_cv", "spend_std",
                   "qty_per_txn", "cust_per_week", "prod_per_txn", "active_weeks"]
    sub = d[hybrid_cols]
    scaled = pd.DataFrame(StandardScaler().fit_transform(sub), columns=hybrid_cols)
    corr = scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    final = [f for f in hybrid_cols if f not in to_drop]
    print(f"  [Hybrid] features kept: {final}")
    return scaled[final].values, final


# D. Baseline with RobustScaler (instead of StandardScaler) to test if
#    outlier-driven scale differences explain the 10-store cluster.
def make_robust_X(df):
    d = df.copy()
    for c in BASELINE_VOL_COLS:
        if c in d.columns:
            d[c] = np.log1p(d[c])
    feat_cols = [c for c in d.columns if c != "STORE_CODE"]
    scaled = pd.DataFrame(RobustScaler().fit_transform(d[feat_cols]), columns=feat_cols)
    corr = scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    final = [f for f in feat_cols if f not in to_drop]
    print(f"  [Robust] features kept: {final}")
    return scaled[final].values, final


# ── Run experiments ─────────────────────────────────────────────────────────
results = []
k_range = range(2, 9)

print("\n── VARIANT A: Baseline features (Log1p + StandardScaler) ──")
X_base, feats_base = make_baseline_X(df)
run_kmeans_and_ward(X_base, k_range, "A_Baseline", "Log1p+StandardScaler", results)
gmm_experiments(X_base, range(2, 6), "A_Baseline", "Log1p+StandardScaler", results)

print("\n── VARIANT B: Ratio/normalized features (Log1p + StandardScaler) ──")
X_ratio, feats_ratio = make_ratio_X(df)
run_kmeans_and_ward(X_ratio, k_range, "B_Ratio", "Log1p+StandardScaler", results)
gmm_experiments(X_ratio, range(2, 6), "B_Ratio", "Log1p+StandardScaler", results)

print("\n── VARIANT C: Hybrid features (rate + volatility + diversity) ──")
X_hybrid, feats_hybrid = make_hybrid_X(df)
run_kmeans_and_ward(X_hybrid, k_range, "C_Hybrid", "Log1p+StandardScaler", results)
gmm_experiments(X_hybrid, range(2, 6), "C_Hybrid", "Log1p+StandardScaler", results)

print("\n── VARIANT D: Baseline features + RobustScaler ──")
X_robust, feats_robust = make_robust_X(df)
run_kmeans_and_ward(X_robust, k_range, "D_Robust", "Log1p+RobustScaler", results)


# ── Save full results ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(EXP_OUT, "experiment_results.csv"), index=False)
print(f"\nTotal experiment rows: {len(results_df)}")


# ── 10-store cluster investigation ─────────────────────────────────────────
print("\n── Investigating the 10-store cluster ──")

# Re-run baseline k=3 to get cluster assignments
km3_base = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_base)
labels3  = km3_base.labels_
df["cluster_base_k3"] = labels3
counts3  = pd.Series(labels3).value_counts().sort_values()
small_cluster_id = counts3.index[0]
print(f"  Small cluster label: {small_cluster_id} | size: {counts3.iloc[0]}")

feat_cols_base = feats_base
cluster_means = df.groupby("cluster_base_k3")[feat_cols_base].mean()

# z-scores relative to overall
overall_mean = df[feat_cols_base].mean()
overall_std  = df[feat_cols_base].std()
z_profiles   = (cluster_means - overall_mean) / overall_std
print("\nZ-score profiles for baseline k=3 clusters:")
print(z_profiles.to_string())

# Save small-cluster profile
small_stores = df[df["cluster_base_k3"] == small_cluster_id]
small_profile = pd.DataFrame({
    "Feature": feat_cols_base,
    "Small_Cluster_Mean": small_stores[feat_cols_base].mean().values,
    "Overall_Mean": overall_mean.values,
    "Z_Score": z_profiles.loc[small_cluster_id].values
})
small_profile.to_csv(os.path.join(EXP_OUT, "small_cluster_profile.csv"), index=False)
print(f"\nSmall cluster z-profile saved.")

# Check ARI stability of k=3 baseline across seeds
seed_labels = []
for seed in SEEDS:
    lbl = KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(X_base)
    seed_labels.append(lbl)

base_aris = []
for lbl in seed_labels:
    base_aris.append(adjusted_rand_score(labels3, lbl))
print(f"\nBaseline k=3 ARI across seeds: {[round(a,4) for a in base_aris]}")
print(f"  Mean ARI: {np.mean(base_aris):.4f}  |  Min: {np.min(base_aris):.4f}")

# Does any ratio variant eliminate the 10-store cluster?
for vname, X_v in [("B_Ratio", X_ratio), ("C_Hybrid", X_hybrid)]:
    km3v = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_v)
    cts  = pd.Series(km3v).value_counts().sort_values()
    print(f"  {vname} k=3 cluster sizes: {list(cts.values)} | min: {cts.min()}")


# ── Select best candidates ─────────────────────────────────────────────────
print("\n── Shortlisting best candidates ──")
# Filter: interpretability OK, reasonable min cluster, good ARI
candidates = results_df[
    (results_df["Interpretability"] == "Acceptable") &
    (results_df["Min_Cluster"] >= 15) &
    (results_df["Mean_ARI"] >= 0.50)
].copy()

print(f"  Candidates passing filters: {len(candidates)}")

# Score: weighted composite (ARI 40%, Silhouette 30%, balance 30%)
if len(candidates) > 0:
    # Normalise each metric to [0,1] over the candidates
    for col, ascending in [("Mean_ARI", False), ("Silhouette", False), ("Largest_%", True)]:
        mn = candidates[col].min(); mx = candidates[col].max()
        rng = mx - mn if mx != mn else 1e-9
        if ascending:
            candidates[f"norm_{col}"] = (mx - candidates[col]) / rng
        else:
            candidates[f"norm_{col}"] = (candidates[col] - mn) / rng

    candidates["composite"] = (
        0.40 * candidates["norm_Mean_ARI"] +
        0.30 * candidates["norm_Silhouette"] +
        0.30 * candidates["norm_Largest_%"]
    )
    shortlist = candidates.sort_values("composite", ascending=False).head(10)
    shortlist.to_csv(os.path.join(EXP_OUT, "shortlisted_candidates.csv"), index=False)
    best = shortlist.iloc[0]
    print("\nTop candidate:")
    print(f"  Variant: {best['Feature_Variant']}  Algo: {best['Algorithm']}  k={best['k']}")
    print(f"  Silhouette: {best['Silhouette']}  DB: {best['DB']}  CH: {best['CH']}")
    print(f"  Mean ARI: {best['Mean_ARI']}  Min ARI: {best['Min_ARI']}")
    print(f"  Distribution: {best['Cluster_Distribution']}")
else:
    best = None
    print("  No candidate passed all filters — baseline retained.")


# ── Generate cluster profiles for best candidate ───────────────────────────
if best is not None and best["Mean_ARI"] > BASELINE["Mean_ARI"] and best["Silhouette"] > BASELINE["Silhouette"] - 0.03:
    print(f"\n  Profiling best candidate: {best['Feature_Variant']} / {best['Algorithm']} / k={best['k']}")
    vmap = {"A_Baseline": X_base, "B_Ratio": X_ratio, "C_Hybrid": X_hybrid, "D_Robust": X_robust}
    X_cand = vmap.get(best["Feature_Variant"], X_base)
    k_cand = int(best["k"])
    algo   = best["Algorithm"]

    if algo == "kmeans":
        cand_labels = KMeans(n_clusters=k_cand, random_state=42, n_init=10).fit_predict(X_cand)
    elif algo == "ward":
        cand_labels = AgglomerativeClustering(n_clusters=k_cand, linkage="ward").fit_predict(X_cand)
    else:
        gm = GaussianMixture(n_components=k_cand, random_state=42, n_init=5)
        cand_labels = gm.fit_predict(X_cand)

    df["cluster_candidate"] = cand_labels
    cts_cand = pd.Series(cand_labels).value_counts().sort_index()
    print(f"  Candidate cluster sizes: {list(cts_cand.values)}")

    feat_map = {
        "A_Baseline": feats_base,
        "B_Ratio": feats_ratio,
        "C_Hybrid": feats_hybrid,
        "D_Robust": feats_robust,
    }
    fts = feat_map.get(best["Feature_Variant"], feats_base)
    # profile using original (pre-scaled) df columns that exist
    orig_cols = [c for c in fts if c in df.columns]
    if orig_cols:
        cand_profile = df.groupby("cluster_candidate")[orig_cols].mean()
        cand_profile["Store_Count"] = df.groupby("cluster_candidate").size()
        cand_profile["Pct"] = (cand_profile["Store_Count"] / len(df) * 100).round(2)
        cand_profile.to_csv(os.path.join(EXP_OUT, "candidate_cluster_profile.csv"))
        print("  Candidate cluster profile saved.")


# ── Print summary ──────────────────────────────────────────────────────────
print("\n\n========================================")
print("ASSESSMENT 2 MODEL IMPROVEMENT STUDY")
print("========================================")
print(f"\nBaseline:")
print(f"  K-Means k=3")
print(f"  Silhouette: {BASELINE['Silhouette']}")
print(f"  DB:         {BASELINE['DB']}")
print(f"  CH:         {BASELINE['CH']}")
print(f"  ARI:        {BASELINE['Mean_ARI']}")
print(f"  Clusters:   {BASELINE['Clusters']}")

n_exp = len(results_df)
print(f"\nExperiments performed: {n_exp}")

if best is not None:
    print(f"\nBest experimental candidate:")
    print(f"  Algorithm:         {best['Algorithm']}")
    print(f"  k:                 {best['k']}")
    print(f"  Feature variant:   {best['Feature_Variant']}")
    print(f"  Preprocessing:     {best['Preprocessing']}")
    print(f"\nMetrics:")
    print(f"  Silhouette:        {best['Silhouette']}")
    print(f"  DB:                {best['DB']}")
    print(f"  CH:                {best['CH']}")
    print(f"  Mean ARI:          {best['Mean_ARI']}")
    print(f"  Cluster dist:      {best['Cluster_Distribution']}")

    sil_better   = best["Silhouette"] > BASELINE["Silhouette"]
    ari_better   = best["Mean_ARI"] > BASELINE["Mean_ARI"]
    bal_better   = best["Min_Cluster"] > 10

    print(f"\nBaseline vs candidate:")
    print(f"  Separation (Silhouette): {'Better' if sil_better else 'Worse or Equal'}")
    print(f"  Stability (Mean ARI):    {'Better' if ari_better else 'Worse or Equal'}")
    print(f"  Balance (min cluster):   {'Better' if bal_better else 'Worse or Equal'}")

    clearly_better = ari_better and bal_better and (sil_better or best["Silhouette"] > BASELINE["Silhouette"] - 0.03)

    if clearly_better:
        print(f"\nRecommendation: RECOMMEND CANDIDATE FOR REVIEW")
        print(f"Reason: The candidate provides better or equivalent separation, improved ARI stability, and better cluster balance than the baseline.")
    else:
        print(f"\nRecommendation: KEEP BASELINE")
        print(f"Reason: The candidate does not improve on all three dimensions simultaneously. Baseline retained.")
else:
    print(f"\nRecommendation: KEEP BASELINE")
    print(f"Reason: No candidate passed the minimum thresholds (ARI >= 0.50, min_cluster >= 15, Interpretability = Acceptable).")

print(f"\nBaseline files modified: NO")
print(f"Dashboard modified:      NO")
print(f"README modified:         NO")
print(f"Reports modified:        NO")
print(f"PPT modified:            NO")
print("========================================")

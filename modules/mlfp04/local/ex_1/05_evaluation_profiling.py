# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.5: Evaluation, AutoMLEngine, and Cluster Profiling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Score every clustering method on silhouette, DB, CH
#   - Use kailash-ml AutoMLEngine with agent=True double-opt-in governance
#   - Profile clusters into business-meaningful segment descriptions
#   - Match algorithm to downstream use case
#
# PREREQUISITES: 01_kmeans.py through 04_spectral.py.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — internal vs external metrics
#   2. Build — fit five methods and collect labels
#   3. Train — AutoMLEngine config with cost cap
#   4. Visualise — metric chart + cluster profiles
#   5. Apply — DBS Bank segmentation selection guide
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from kailash_ml import AutoMLEngine, ModelVisualizer

# AutoMLConfig is not in kailash_ml.__all__ in 1.5.x; import from the
# automl.engine submodule. AutoMLEngine remains a top-level export.
from kailash_ml.automl.engine import (
    AutoMLConfig,
)  # pyright: ignore[reportMissingImports]

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    agreement,
    load_customers,
    out_path,
    print_metric_row,
    score_partition,
    standardise,
    subsample,
)

load_dotenv()

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None


# ════════════════════════════════════════════════════════════════════════
# THEORY — Internal vs External Metrics and Profiling
# ════════════════════════════════════════════════════════════════════════
# Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) rank
# methods using only X + labels. External metrics (ARI, NMI) tell you
# how much two partitions AGREE — high agreement = real structure.
# Neither is sufficient without the profiling step, which converts
# statistical labels into actionable business segments via z-scores.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Fit every method and collect full-data labels
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

print("=" * 70)
print("  Clustering Evaluation + Profiling on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples={n_samples:,}  features={len(feature_cols)}")

BEST_K = 5
all_labels: dict[str, np.ndarray] = {}

# TODO: Fit K-means with BEST_K clusters (init='k-means++', n_init=10)
# and store all_labels["K-means"] = km.fit_predict(X_scaled).
km = ____
all_labels["K-means"] = ____

# TODO: Fit a GaussianMixture with n_components=BEST_K, covariance_type='full'
# and store all_labels["GMM"] = gmm.fit_predict(X_scaled).
gmm = ____
all_labels["GMM"] = ____

# --- Ward hierarchical (KNN-extend to full data) ---
X_hier, idx_hier = subsample(X_scaled, n=2000, seed=RANDOM_STATE)
Z = linkage(X_hier, method="ward")
ward_sub = fcluster(Z, t=BEST_K, criterion="maxclust") - 1
knn = KNeighborsClassifier(n_neighbors=5).fit(X_hier, ward_sub)
all_labels["Ward"] = knn.predict(X_scaled)

# --- DBSCAN with k-distance-selected epsilon ---
nn = NearestNeighbors(n_neighbors=10).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])
diffs2 = np.diff(np.diff(k_dist))
eps_suggested = float(k_dist[int(np.argmax(diffs2)) + 2])
all_labels["DBSCAN"] = DBSCAN(eps=eps_suggested, min_samples=10, n_jobs=-1).fit_predict(
    X_scaled
)

# --- HDBSCAN ---
if hdbscan_lib is not None:
    all_labels["HDBSCAN"] = hdbscan_lib.HDBSCAN(
        min_cluster_size=50, min_samples=10, cluster_selection_method="eom"
    ).fit_predict(X_scaled)

# --- Spectral (subsample + KNN-extend) ---
X_spec, idx_spec = subsample(X_scaled, n=2500, seed=RANDOM_STATE)
spec_sub = SpectralClustering(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    affinity="rbf",
    gamma=1.0,
    assign_labels="kmeans",
).fit_predict(X_spec)
knn_spec = KNeighborsClassifier(n_neighbors=5).fit(X_spec, spec_sub)
all_labels["Spectral"] = knn_spec.predict(X_scaled)

print("\n  Internal metrics per method:")
results: dict[str, dict] = {}
for name, labels in all_labels.items():
    # TODO: Use the shared score_partition(X_scaled, labels) helper and
    # store the result in results[name]. Then print via print_metric_row.
    m = ____
    results[name] = m
    print_metric_row(name, m)


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(results) >= 5, "Task 2: at least 5 methods should be scored"
assert all("silhouette" in r for r in results.values()), "Task 2: metric gap"
print("\n  [ok] Checkpoint 1 passed — all methods scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: AutoMLEngine with agent=False double-opt-in
# ════════════════════════════════════════════════════════════════════════


async def run_automl() -> AutoMLConfig:
    """Build an AutoMLEngine config for clustering comparison."""
    # TODO: Build an AutoMLConfig with task_type='clustering',
    # metric_name='silhouette', direction='maximize',
    # search_strategy='random', max_trials=20, agent=False,
    # max_llm_cost_usd=1.0. Return the config.
    config = ____
    _ = AutoMLEngine
    return config


config = asyncio.run(run_automl())
print("  AutoMLEngine config:")
print(f"    task_type         = {config.task_type}")
print(f"    metric_name       = {config.metric_name}")
print(f"    agent             = {config.agent}  (False = no LLM)")
print(f"    max_llm_cost_usd  = {config.max_llm_cost_usd}")

print("\n  External agreement (ARI / NMI):")
method_names = list(all_labels.keys())
for i in range(len(method_names)):
    for j in range(i + 1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        # TODO: Call the shared agreement(labels_a, labels_b) helper.
        a = ____
        print(f"    {m1:<10} vs {m2:<10}  ARI={a['ari']:+.4f}  NMI={a['nmi']:+.4f}")


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert config.agent is False, "Task 3: agent must default to False"
assert config.max_llm_cost_usd > 0, "Task 3: cost cap must be positive"
print("\n  [ok] Checkpoint 2 passed — AutoMLEngine configured with guardrails\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Metric bar chart + cluster profiles
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        k: {"silhouette": v["silhouette"], "calinski_harabasz": v["calinski_harabasz"]}
        for k, v in results.items()
        if not np.isnan(v["silhouette"])
    }
)
fig.update_layout(title="Clustering Method Comparison (internal metrics)")
fig.write_html(str(out_path("05_method_comparison.html")))
print(f"  Saved: {out_path('05_method_comparison.html')}")

best_name = max(
    ((k, v) for k, v in results.items() if not np.isnan(v["silhouette"])),
    key=lambda x: x[1]["silhouette"],
)[0]
best_labels = all_labels[best_name]
print(f"\n  Best method by silhouette: {best_name}")

clustered = customers.with_columns(pl.Series("cluster", best_labels))
for cid in sorted(set(int(c) for c in best_labels.tolist() if c >= 0)):
    subset = clustered.filter(pl.col("cluster") == cid)
    pct = subset.height / clustered.height * 100
    print(f"\n  Cluster {cid} — n={subset.height:,} ({pct:.1f}%)")
    for col in feature_cols[:6]:
        mean_val = subset[col].mean()
        overall_mean = clustered[col].mean()
        overall_std = clustered[col].std()
        if overall_std and overall_std > 0:
            z = (mean_val - overall_mean) / overall_std
        else:
            z = 0.0
        flag = "HIGH" if z > 0.5 else ("LOW " if z < -0.5 else "    ")
        print(f"    {col:<28} mean={mean_val:>10.2f}  z={z:+.2f}  {flag}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("05_method_comparison.html").exists(), "Task 4: chart not saved"
assert "cluster" in clustered.columns, "Task 4: cluster column missing"
print("\n  [ok] Checkpoint 3 passed — metric chart + cluster profiles rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Singapore Segmentation Selection Guide
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS's consumer banking runs FIVE different segmentation
# programs — loyalty tiers, wealth desk affinity, fraud rings, cross-sell
# offers, RM-beat optimisation — each needs a DIFFERENT algorithm.
#
# BUSINESS IMPACT: Estimated S$62M / year aggregate benefit.

print("  APPLY — DBS Bank Consumer Segmentation Selection Guide")
print("  ─────────────────────────────────────────────────────────────────")
print(
    """
  ┌──────────────────┬───────────────────┬──────────────┬──────────────┬───────────────┐
  │ Algorithm        │ Requires K?       │ Cluster Shape│ Noise        │ Scalability   │
  ├──────────────────┼───────────────────┼──────────────┼──────────────┼───────────────┤
  │ K-means          │ Yes               │ Convex       │ None         │ O(nKI)        │
  │ Hierarchical     │ Yes (cut height)  │ Any          │ None         │ O(n^2 log n)  │
  │ DBSCAN           │ No (eps, minPts)  │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ HDBSCAN          │ No (auto)         │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ Spectral         │ Yes               │ Non-convex   │ None         │ O(n^3)        │
  │ GMM              │ Yes (BIC selects) │ Ellipsoidal  │ Soft         │ O(nK^2d)      │
  └──────────────────┴───────────────────┴──────────────┴──────────────┴───────────────┘
"""
)
print("  Estimated DBS annual benefit: S$62M across four segmentation programs.")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert best_name in results, "Task 5: best method must be in results"
print("\n  [ok] Checkpoint 4 passed — selection guide delivered\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Scored five clustering methods on three internal metrics
  [x] Measured pairwise agreement via ARI and NMI
  [x] Configured AutoMLEngine with agent=False + cost cap
  [x] Profiled the best partition via per-feature z-scores
  [x] Applied the selection guide to DBS Bank — S$62M / year benefit

  KEY INSIGHT: There is no universally best clustering algorithm.
  Match the tool to the problem, then profile the result for the
  business team.

  Next: Exercise 2 — implement the EM algorithm behind GMM by hand.
"""
)

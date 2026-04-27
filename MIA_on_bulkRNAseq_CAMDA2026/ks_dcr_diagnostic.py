#!/usr/bin/env python3
"""
ks_dcr_diagnostic.py
--------------------
Standalone diagnostic for CAMDA26-KIMJONA.

Run from inside  MIA_on_bulkRNAseq_CAMDA2026/  (same dir as the other scripts):

    python ks_dcr_diagnostic.py --dataset BRCA
    python ks_dcr_diagnostic.py --dataset COMBINED
    python ks_dcr_diagnostic.py --dataset both

Output written to  mia_output/diagnostics/<dataset>/
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from mia import config
from mia.data_utils import (
    load_real_data,
    load_nd_synthetic,
    get_nd_membership_labels,
    fit_quantile_scaler,
)

# ── tuneable constants ────────────────────────────────────────────────────────
KS_D_THRESHOLD   = 0.10   # genes with D < this are "well-represented"
PCA_N_COMPONENTS = 50     # dimensionality for DCR (avoids curse of dimensionality)
DCR_N_SUBSAMPLE  = 2000   # cap synthetic points for pairwise distance (speed)


# ── KS fidelity ───────────────────────────────────────────────────────────────

def compute_ks_fidelity(X_real, X_synth, d_threshold=KS_D_THRESHOLD):
    """
    Per-gene KS D-statistic only.  p-values are intentionally discarded:
    at n > 3000 every gene will fail p < 0.05 even when D is tiny.
    D = 0 → identical CDFs.  D = 1 → completely disjoint distributions.
    """
    n_genes = X_real.shape[1]
    stats   = np.empty(n_genes, dtype=np.float32)
    for g in range(n_genes):
        d, _ = ks_2samp(X_real[:, g], X_synth[:, g])
        stats[g] = d
    return {
        "mean_D":            float(np.mean(stats)),
        "median_D":          float(np.median(stats)),
        "p90_D":             float(np.percentile(stats, 90)),
        "frac_below_thresh": float(np.mean(stats < d_threshold)),
        "per_gene_D":        stats,
    }


# ── DCR bound ─────────────────────────────────────────────────────────────────

def _pairwise_l2(A, B):
    """Efficient ||A_i - B_j||_2 matrix via broadcasting."""
    A2 = np.sum(A ** 2, axis=1, keepdims=True)
    B2 = np.sum(B ** 2, axis=1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2 * (A @ B.T), 0.0)
    return np.sqrt(D2)


def compute_dcr(X_syn_pca, X_real_pca, y_member, n_subsample=DCR_N_SUBSAMPLE, seed=config.SEED):
    """
    For each SYNTHETIC sample compute:
      d_mem = distance to nearest REAL member (training set)
      d_non = distance to nearest REAL non-member (held-out set)

    Scaler and PCA are fit on real data only, then used to project synthetic
    into the same geometric space — no leakage from synthetic into the fit.

    ratio = d_mem / d_non
      << 1.0  →  Synthetic points spawn on top of members → memorisation exists
      ≈  1.0  →  Synthetic equidistant to seen/unseen
    """
    rng         = np.random.default_rng(seed)
    members     = X_real_pca[y_member == 1]
    non_members = X_real_pca[y_member == 0]

    # FIX: Balance pool sizes so the min-distance bias is symmetric
    k = min(len(members), len(non_members))
    if len(members) > k:
        members = members[rng.choice(len(members), k, replace=False)]
    if len(non_members) > k:
        non_members = non_members[rng.choice(len(non_members), k, replace=False)]

    # Subsample synthetic points for speed
    if n_subsample and len(X_syn_pca) > n_subsample:
        X_syn_pca = X_syn_pca[rng.choice(len(X_syn_pca), n_subsample, replace=False)]

    # Distance from SYNTHETIC → REAL members / non-members
    d_mem = _pairwise_l2(X_syn_pca, members).min(axis=1)
    d_non = _pairwise_l2(X_syn_pca, non_members).min(axis=1)
    ratio = d_mem / np.maximum(d_non, 1e-12)

    return {
        "d_mem_mean":   float(np.mean(d_mem)),
        "d_non_mean":   float(np.mean(d_non)),
        "ratio_mean":   float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "d_mem":        d_mem,
        "d_non":        d_non,
        "ratio":        ratio,
    }


def _interpret_ratio(r):
    if r < 0.50:
        return "HIGH memorisation  — synthetic clusters on members"
    if r < 0.85:
        return "MODERATE           — partial clustering, MIA leakage plausible"
    return  "LOW memorisation   — synthetic equidistant to seen/unseen"


# ── main ──────────────────────────────────────────────────────────────────────

def run(dataset_name):
    out_dir = os.path.join(config.MIA_OUTPUT_DIR, "diagnostics", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*65}")

    X_real_df, _ = load_real_data(dataset_name)
    X_real        = X_real_df.values.astype(np.float32)
    print(f"  Real data shape: {X_real.shape}")

    n_splits = config.NUM_SPLITS

    # ── KS per split ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  KS Fidelity  (D-statistic only, threshold={KS_D_THRESHOLD})")
    print(f"  {'Split':>6}  {'mean_D':>8}  {'median_D':>9}  {'p90_D':>7}  {'frac<thr':>9}")
    print(f"{'─'*65}")

    ks_rows = []
    for s in range(1, n_splits + 1):
        X_syn, _ = load_nd_synthetic(dataset_name, s)
        res      = compute_ks_fidelity(X_real, X_syn)
        print(f"  {s:>6}  {res['mean_D']:>8.4f}  {res['median_D']:>9.4f}  "
              f"{res['p90_D']:>7.4f}  {res['frac_below_thresh']:>9.4f}")
        ks_rows.append({
            "split":             s,
            "mean_D":            res["mean_D"],
            "median_D":          res["median_D"],
            "p90_D":             res["p90_D"],
            "frac_below_thresh": res["frac_below_thresh"],
        })
        np.save(os.path.join(out_dir, f"ks_per_gene_split_{s}.npy"), res["per_gene_D"])

    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(os.path.join(out_dir, "ks_summary.csv"), index=False)
    print(f"\n  → {out_dir}/ks_summary.csv")
    print(f"  → {out_dir}/ks_per_gene_split_*.npy")

    # ── DCR per split ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  DCR Bound  (Synthetic → Real, PCA={PCA_N_COMPONENTS}D, subsample={DCR_N_SUBSAMPLE})")
    print(f"  ratio = d_mem / d_non  (ideal ≈ 1.0 → no memorisation)")
    print(f"  {'Split':>6}  {'d_mem':>8}  {'d_non':>8}  {'ratio':>7}  Interpretation")
    print(f"{'─'*65}")

    # Fit scaler and PCA ONCE on real data only — synthetic is projected in,
    # never used to fit, so there is no leakage into the geometric space.
    scaler, X_real_scaled = fit_quantile_scaler(X_real)
    pca                   = PCA(n_components=PCA_N_COMPONENTS, random_state=config.SEED)
    X_real_pca            = pca.fit_transform(X_real_scaled)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    dcr_rows = []
    for s in range(1, n_splits + 1):
        # Load synthetic for this split
        X_syn, _     = load_nd_synthetic(dataset_name, s)

        # Project synthetic into the same geometric space as real data
        X_syn_scaled = scaler.transform(X_syn.astype(np.float64)).astype(np.float32)
        X_syn_pca    = pca.transform(X_syn_scaled)

        # Membership labels: who was in the training set for this split
        _, y_member  = get_nd_membership_labels(dataset_name, s)

        # DCR: Synthetic → Real (the correct direction)
        res = compute_dcr(X_syn_pca, X_real_pca, y_member)

        print(f"  {s:>6}  {res['d_mem_mean']:>8.4f}  {res['d_non_mean']:>8.4f}  "
              f"{res['ratio_mean']:>7.4f}  {_interpret_ratio(res['ratio_mean'])}")
        dcr_rows.append({
            "split":        s,
            "d_mem_mean":   res["d_mem_mean"],
            "d_non_mean":   res["d_non_mean"],
            "ratio_mean":   res["ratio_mean"],
            "ratio_median": res["ratio_median"],
        })
        np.save(os.path.join(out_dir, f"dcr_ratio_split_{s}.npy"), res["ratio"])

    dcr_df = pd.DataFrame(dcr_rows)
    dcr_df.to_csv(os.path.join(out_dir, "dcr_summary.csv"), index=False)
    print(f"\n  → {out_dir}/dcr_summary.csv")
    print(f"  → {out_dir}/dcr_ratio_split_*.npy")

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_frac  = float(ks_df["frac_below_thresh"].mean())
    mean_ratio = float(dcr_df["ratio_mean"].mean())

    print(f"\n{'='*65}")
    print(f"  SUMMARY — {dataset_name}")
    print(f"{'='*65}")
    print(f"  KS  avg fraction of genes with D < {KS_D_THRESHOLD}:  {mean_frac:.3f}")
    print(f"       → {'Generator learned the marginal distributions' if mean_frac > 0.6 else 'Generator underfit features'}")
    print(f"\n  DCR avg ratio (synthetic → real):         {mean_ratio:.3f}")
    print(f"       → {_interpret_ratio(mean_ratio)}")
    print(f"\n  NOTE: KS is a diagnostic only — not an MIA bound.")
    print(f"        DCR ratio is a geometric indicator of member-neighbourhood memorisation.")
    print(f"        Values near 1.0 suggest the generator hasn't collapsed onto training points,")
    print(f"        but do not preclude distributional MIA leakage.")
    print(f"        LOSO-CV AUC remains the empirical upper bound for your attack architecture.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED", "both"], default="both")
    parser.add_argument("--profile", choices=list(config.PROFILES.keys()), default=None)
    args = parser.parse_args()

    if args.profile:
        config.apply_profile(args.profile)

    datasets = ["BRCA", "COMBINED"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run(ds)
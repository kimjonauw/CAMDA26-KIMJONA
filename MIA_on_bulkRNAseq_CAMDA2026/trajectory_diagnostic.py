"""Diagnostic: visualize member vs non-member loss trajectories to identify
the exact geometric difference the current summary statistics miss.

Outputs:
  1. Mean loss trajectories (member vs non-member) with confidence bands
  2. Per-noise-vector std trajectories (member vs non-member)
  3. Individual trajectory overlays (random sample of 50 members + 50 non-members)
  4. Distribution of per-sample trajectory shapes (derivative profiles)
  5. Correlation matrix of current summary features
  6. Feature-by-feature member/non-member distribution comparison
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew as scipy_skew
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia import config
from mia.loss_features import slice_raw_features, summarize_features

OUT_DIR = os.path.join(config.MIA_OUTPUT_DIR, "diagnostics", "trajectory_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

def load_split_data(dataset_name, split_no):
    feat_dir = os.path.join(config.MIA_OUTPUT_DIR, "synth_shadow", "features", dataset_name)
    d = np.load(os.path.join(feat_dir, f"features_split_{split_no}.npz"))
    raw = d["features"]  # (n_samples, n_t * n_noise) — UNCALIBRATED
    y = d["y_member"]
    ids = d["sample_ids"]
    ref = d["ref_features"] if "ref_features" in d else None
    return raw, y, ids, ref

def reshape_to_3d(raw, n_t, n_noise):
    """(n_samples, n_t*n_noise) -> (n_samples, n_t, n_noise)"""
    return raw.reshape(raw.shape[0], n_t, n_noise)

def plot_mean_trajectories(members_3d, nonmembers_3d, t_list, dataset_name):
    """Plot 1: Mean loss trajectory with std bands for members vs non-members."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-sample mean across noise vectors -> (n_samples, n_t)
    mem_means = members_3d.mean(axis=2)
    non_means = nonmembers_3d.mean(axis=2)
    
    # Population mean and std
    mem_pop_mean = mem_means.mean(axis=0)
    mem_pop_std = mem_means.std(axis=0)
    non_pop_mean = non_means.mean(axis=0)
    non_pop_std = non_means.std(axis=0)
    
    ax = axes[0]
    x = np.arange(len(t_list))
    ax.plot(x, mem_pop_mean, 'b-o', label='Members', linewidth=2, markersize=4)
    ax.fill_between(x, mem_pop_mean - mem_pop_std, mem_pop_mean + mem_pop_std, alpha=0.15, color='blue')
    ax.plot(x, non_pop_mean, 'r-s', label='Non-members', linewidth=2, markersize=4)
    ax.fill_between(x, non_pop_mean - non_pop_std, non_pop_mean + non_pop_std, alpha=0.15, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Mean loss (across noise vectors)")
    ax.set_title(f"{dataset_name}: Mean Loss Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot the DIFFERENCE
    ax = axes[1]
    diff = non_pop_mean - mem_pop_mean
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Non-member mean - Member mean")
    ax.set_title(f"{dataset_name}: Mean Loss Difference (Non-member − Member)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"mean_trajectories_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return mem_means, non_means

def plot_std_trajectories(members_3d, nonmembers_3d, t_list, dataset_name):
    """Plot 2: Per-sample std across noise vectors — the variance signal."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-sample std across noise vectors -> (n_samples, n_t)
    mem_stds = members_3d.std(axis=2)
    non_stds = nonmembers_3d.std(axis=2)
    
    mem_pop_mean = mem_stds.mean(axis=0)
    mem_pop_std = mem_stds.std(axis=0)
    non_pop_mean = non_stds.mean(axis=0)
    non_pop_std = non_stds.std(axis=0)
    
    ax = axes[0]
    x = np.arange(len(t_list))
    ax.plot(x, mem_pop_mean, 'b-o', label='Members', linewidth=2, markersize=4)
    ax.fill_between(x, mem_pop_mean - mem_pop_std, mem_pop_mean + mem_pop_std, alpha=0.15, color='blue')
    ax.plot(x, non_pop_mean, 'r-s', label='Non-members', linewidth=2, markersize=4)
    ax.fill_between(x, non_pop_mean - non_pop_std, non_pop_mean + non_pop_std, alpha=0.15, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Std of loss (across noise vectors)")
    ax.set_title(f"{dataset_name}: Loss Variance Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference
    ax = axes[1]
    diff = non_pop_mean - mem_pop_mean
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Non-member std - Member std")
    ax.set_title(f"{dataset_name}: Loss Variance Difference")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"std_trajectories_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return mem_stds, non_stds

def plot_individual_trajectories(members_3d, nonmembers_3d, t_list, dataset_name, n_show=30):
    """Plot 3: Overlay individual sample trajectories (mean across noise)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(t_list))
    
    rng = np.random.RandomState(42)
    
    mem_means = members_3d.mean(axis=2)
    non_means = nonmembers_3d.mean(axis=2)
    
    n_mem = min(n_show, len(mem_means))
    n_non = min(n_show, len(non_means))
    
    ax = axes[0]
    idx = rng.choice(len(mem_means), n_mem, replace=False)
    for i in idx:
        ax.plot(x, mem_means[i], 'b-', alpha=0.15, linewidth=0.8)
    ax.plot(x, mem_means.mean(axis=0), 'b-', linewidth=3, label='Population mean')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_title(f"{dataset_name}: Individual MEMBER trajectories (n={n_mem})")
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Mean loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    idx = rng.choice(len(non_means), n_non, replace=False)
    for i in idx:
        ax.plot(x, non_means[i], 'r-', alpha=0.15, linewidth=0.8)
    ax.plot(x, non_means.mean(axis=0), 'r-', linewidth=3, label='Population mean')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_title(f"{dataset_name}: Individual NON-MEMBER trajectories (n={n_non})")
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Mean loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Match y-axis
    ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin, ymax)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"individual_trajectories_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_derivative_profiles(mem_means, non_means, t_list, dataset_name):
    """Plot 4: First and second derivative distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # First derivative (between consecutive timesteps)
    mem_grad1 = np.diff(mem_means, axis=1)
    non_grad1 = np.diff(non_means, axis=1)
    
    # Second derivative
    mem_grad2 = np.diff(mem_means, n=2, axis=1)
    non_grad2 = np.diff(non_means, n=2, axis=1)
    
    # Plot 4a: Mean first derivative trajectory
    ax = axes[0, 0]
    x = np.arange(len(t_list) - 1)
    labels = [f"{t_list[i]}→{t_list[i+1]}" for i in range(len(t_list)-1)]
    ax.plot(x, mem_grad1.mean(axis=0), 'b-o', label='Members', linewidth=2, markersize=4)
    ax.plot(x, non_grad1.mean(axis=0), 'r-s', label='Non-members', linewidth=2, markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=6)
    ax.set_title("First Derivative (Δ loss between adjacent t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4b: Max absolute first derivative per sample (steepest slope)
    ax = axes[0, 1]
    mem_max_deriv = np.max(np.abs(mem_grad1), axis=1)
    non_max_deriv = np.max(np.abs(non_grad1), axis=1)
    ax.hist(mem_max_deriv, bins=50, alpha=0.5, color='blue', label=f'Members (mean={mem_max_deriv.mean():.4f})', density=True)
    ax.hist(non_max_deriv, bins=50, alpha=0.5, color='red', label=f'Non-members (mean={non_max_deriv.mean():.4f})', density=True)
    ax.set_title("Max |first derivative| per sample (steepest slope)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4c: Mean second derivative trajectory  
    ax = axes[1, 0]
    x2 = np.arange(len(t_list) - 2)
    labels2 = [f"{t_list[i]}-{t_list[i+1]}-{t_list[i+2]}" for i in range(len(t_list)-2)]
    ax.plot(x2, mem_grad2.mean(axis=0), 'b-o', label='Members', linewidth=2, markersize=4)
    ax.plot(x2, non_grad2.mean(axis=0), 'r-s', label='Non-members', linewidth=2, markersize=4)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, rotation=60, fontsize=5)
    ax.set_title("Second Derivative (curvature)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4d: Trajectory roughness (variance of second derivative per sample)
    ax = axes[1, 1]
    mem_roughness = np.var(mem_grad2, axis=1)
    non_roughness = np.var(non_grad2, axis=1)
    ax.hist(mem_roughness, bins=50, alpha=0.5, color='blue', label=f'Members (mean={mem_roughness.mean():.6f})', density=True)
    ax.hist(non_roughness, bins=50, alpha=0.5, color='red', label=f'Non-members (mean={non_roughness.mean():.6f})', density=True)
    ax.set_title("Trajectory roughness (var of 2nd derivative)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"{dataset_name}: Derivative Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"derivative_profiles_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return mem_grad1, non_grad1, mem_grad2, non_grad2

def plot_timestep_of_min_loss(mem_means, non_means, t_list, dataset_name):
    """Plot 5: Where in the trajectory does each sample's minimum loss occur?"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    mem_argmin = np.argmin(mem_means, axis=1)
    non_argmin = np.argmin(non_means, axis=1)
    
    bins = np.arange(-0.5, len(t_list) + 0.5, 1)
    ax.hist(mem_argmin, bins=bins, alpha=0.5, color='blue', label='Members', density=True)
    ax.hist(non_argmin, bins=bins, alpha=0.5, color='red', label='Non-members', density=True)
    ax.set_xticks(range(len(t_list)))
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_xlabel("Timestep of minimum mean loss")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name}: Timestep of Minimum Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"argmin_loss_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_noise_consistency(members_3d, nonmembers_3d, t_list, dataset_name):
    """Plot 6: For each sample, how consistent are the losses across noise vectors?
    Hypothesis: members have MORE consistent (lower entropy) loss distributions at each t."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coefficient of variation: std/mean per (sample, timestep)
    mem_cv = members_3d.std(axis=2) / (np.abs(members_3d.mean(axis=2)) + 1e-8)
    non_cv = nonmembers_3d.std(axis=2) / (np.abs(nonmembers_3d.mean(axis=2)) + 1e-8)
    
    ax = axes[0]
    x = np.arange(len(t_list))
    ax.plot(x, mem_cv.mean(axis=0), 'b-o', label='Members', linewidth=2, markersize=4)
    ax.fill_between(x, mem_cv.mean(0) - mem_cv.std(0), mem_cv.mean(0) + mem_cv.std(0), alpha=0.15, color='blue')
    ax.plot(x, non_cv.mean(axis=0), 'r-s', label='Non-members', linewidth=2, markersize=4)
    ax.fill_between(x, non_cv.mean(0) - non_cv.std(0), non_cv.mean(0) + non_cv.std(0), alpha=0.15, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_title(f"{dataset_name}: Coefficient of Variation (std/|mean|)")
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("CV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference in CV
    ax = axes[1]
    diff = non_cv.mean(axis=0) - mem_cv.mean(axis=0)
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_list], rotation=45, fontsize=8)
    ax.set_title(f"{dataset_name}: CV Difference (Non-member − Member)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"noise_consistency_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_correlation_matrix(raw, y, t_list, n_noise, dataset_name):
    """Plot 7: Correlation matrix of current summary features."""
    n_t = len(t_list)
    t_indices = list(range(n_t))
    
    X = summarize_features(slice_raw_features(raw, t_indices, n_noise), n_noise, n_t)
    
    # Build feature names
    stat_names = ["mean", "std", "min", "max", "median", "skew"]
    feat_names = []
    for stat in stat_names:
        for t in t_list:
            feat_names.append(f"{stat}_T{t}")
    for i in range(n_t - 1):
        feat_names.append(f"grad_{t_list[i]}→{t_list[i+1]}")
    
    corr = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 18))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, xticklabels=feat_names, yticklabels=feat_names,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.1, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{dataset_name}: Feature Correlation Matrix", fontsize=14)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"correlation_matrix_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    # Print highly correlated pairs (>0.95)
    print(f"\n  Pairs with |correlation| > 0.95 ({dataset_name}):")
    count = 0
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            if abs(corr[i, j]) > 0.95:
                print(f"    {feat_names[i]:20s} ↔ {feat_names[j]:20s}  r={corr[i,j]:.4f}")
                count += 1
    print(f"  Total highly correlated pairs: {count}")
    
    return X, feat_names, corr

def plot_member_nonmember_feature_separation(X, y, feat_names, dataset_name):
    """Plot 8: For each feature, how well does it separate members from non-members?
    Measured by |mean_member - mean_nonmember| / pooled_std (Cohen's d)."""
    mem_mask = y == 1
    non_mask = y == 0
    
    X_mem = X[mem_mask]
    X_non = X[non_mask]
    
    cohens_d = []
    for j in range(X.shape[1]):
        m1, m0 = X_mem[:, j].mean(), X_non[:, j].mean()
        s1, s0 = X_mem[:, j].std(), X_non[:, j].std()
        pooled = np.sqrt((s1**2 + s0**2) / 2)
        d = (m1 - m0) / (pooled + 1e-8)
        cohens_d.append(d)
    
    cohens_d = np.array(cohens_d)
    sorted_idx = np.argsort(np.abs(cohens_d))[::-1]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    top_k = min(40, len(feat_names))
    idx = sorted_idx[:top_k]
    names = [feat_names[i] for i in idx]
    vals = [cohens_d[i] for i in idx]
    colors = ['blue' if v < 0 else 'red' for v in vals]  # negative = members lower
    
    ax.barh(range(top_k), vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Cohen's d (negative = members have LOWER values)")
    ax.set_title(f"{dataset_name}: Feature Separation Power (Cohen's d)")
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"cohens_d_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    print(f"\n  Top 15 features by |Cohen's d| ({dataset_name}):")
    for rank, i in enumerate(sorted_idx[:15]):
        print(f"    {rank+1:2d}. {feat_names[i]:20s}  d={cohens_d[i]:+.4f}")

def run_for_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"  TRAJECTORY DIAGNOSTIC: {dataset_name}")
    print(f"{'='*60}")
    
    n_t = len(config.T_SUPERSET)
    n_noise = config.N_NOISE
    
    # Load first split as representative
    raw, y, ids, ref = load_split_data(dataset_name, 1)
    print(f"  Raw shape: {raw.shape}, n_members={int(y.sum())}, n_nonmembers={int((y==0).sum())}")
    
    raw_3d = reshape_to_3d(raw, n_t, n_noise)
    
    mem_mask = y == 1
    non_mask = y == 0
    members_3d = raw_3d[mem_mask]
    nonmembers_3d = raw_3d[non_mask]
    
    print(f"  Members 3D: {members_3d.shape}, Non-members 3D: {nonmembers_3d.shape}")
    
    # 1. Mean trajectories
    mem_means, non_means = plot_mean_trajectories(members_3d, nonmembers_3d, config.T_SUPERSET, dataset_name)
    
    # 2. Std trajectories
    mem_stds, non_stds = plot_std_trajectories(members_3d, nonmembers_3d, config.T_SUPERSET, dataset_name)
    
    # 3. Individual trajectories
    plot_individual_trajectories(members_3d, nonmembers_3d, config.T_SUPERSET, dataset_name)
    
    # 4. Derivative profiles
    plot_derivative_profiles(mem_means, non_means, config.T_SUPERSET, dataset_name)
    
    # 5. Timestep of minimum loss
    plot_timestep_of_min_loss(mem_means, non_means, config.T_SUPERSET, dataset_name)
    
    # 6. Noise consistency
    plot_noise_consistency(members_3d, nonmembers_3d, config.T_SUPERSET, dataset_name)
    
    # 7. Correlation matrix
    X, feat_names, corr = plot_correlation_matrix(raw, y, config.T_SUPERSET, n_noise, dataset_name)
    
    # 8. Feature separation
    plot_member_nonmember_feature_separation(X, y, feat_names, dataset_name)

if __name__ == "__main__":
    for ds in ["BRCA", "COMBINED"]:
        try:
            run_for_dataset(ds)
        except Exception as e:
            print(f"  ERROR for {ds}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\nAll diagnostics saved to: {OUT_DIR}")

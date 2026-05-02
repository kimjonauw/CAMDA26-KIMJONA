import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia import config
from mia.loss_features import slice_raw_features, summarize_features

def load_split_data(dataset_name, split_no):
    feat_dir = os.path.join(config.MIA_OUTPUT_DIR, "synth_shadow", "features", dataset_name)
    d = np.load(os.path.join(feat_dir, f"features_split_{split_no}.npz"))
    raw = d["features"]
    y = d["y_member"]
    return raw, y

def print_stats(dataset_name):
    print(f"--- {dataset_name} ---")
    raw, y = load_split_data(dataset_name, 1)
    
    n_t = len(config.T_SUPERSET)
    n_noise = config.N_NOISE
    t_list = config.T_SUPERSET
    t_indices = list(range(n_t))
    
    X = summarize_features(slice_raw_features(raw, t_indices, n_noise), n_noise, n_t)
    
    stat_names = ["mean", "std", "min", "max", "median", "skew"]
    feat_names = []
    for stat in stat_names:
        for t in t_list:
            feat_names.append(f"{stat}_T{t}")
    for i in range(n_t - 1):
        feat_names.append(f"grad_{t_list[i]}->{t_list[i+1]}")
        
    corr = np.corrcoef(X.T)
    count = 0
    print(f"\nHighly correlated pairs (|r| > 0.95):")
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            if abs(corr[i, j]) > 0.95:
                print(f"  {feat_names[i]:20s} <-> {feat_names[j]:20s}  r={corr[i,j]:.4f}")
                count += 1
    print(f"Total highly correlated pairs: {count}")
    
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
    
    print(f"\nTop 20 features by |Cohen's d|:")
    for rank, i in enumerate(sorted_idx[:20]):
        print(f"  {rank+1:2d}. {feat_names[i]:20s}  d={cohens_d[i]:+.4f}")
    print("\n")

if __name__ == "__main__":
    for ds in ["BRCA", "COMBINED"]:
        print_stats(ds)

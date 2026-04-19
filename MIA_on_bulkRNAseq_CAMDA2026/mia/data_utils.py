"""Data loading and preprocessing for the MIA pipeline.

The challenge provides:
  - Real data TSV: genes × samples (rows=978 genes, columns=sample IDs). Must transpose.
  - Synthetic data CSV: samples × genes (rows=samples, columns=gene IDs). No labels.

The NoisyDiffusion repo provides:
  - Labeled synthetic data: synthetic_data_split_{k}.csv + synthetic_labels_split_{k}.csv
  - Splits YAML: which sample IDs are train/test per fold → ground truth membership.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier

from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Loading challenge data
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(dataset_name):
    """Load the full real-data TSV (genes × samples), transpose to (samples × genes).

    Returns:
        X : pd.DataFrame, shape (n_samples, 978), index = sample IDs
        gene_names : list[str], the 978 gene ENSEMBL IDs
    """
    ds = config.DATASETS[dataset_name]
    df = pd.read_csv(ds["real_tsv"], sep="\t", index_col=0)
    # df is (978 genes × n_samples) → transpose to (n_samples × 978)
    X = df.T
    gene_names = list(df.index)
    return X, gene_names


def load_challenge_synthetic(dataset_name):
    """Load the challenge's synthetic CSV (samples × genes, no labels).

    Returns:
        X : np.ndarray, shape (n_syn, 978)
    """
    ds = config.DATASETS[dataset_name]
    df = pd.read_csv(ds["synthetic_csv"])
    return df.values.astype(np.float32)


def load_reference_data(dataset_name):
    """Load the reference dataset (COMBINED only). Same format as real data TSV.

    Returns:
        X : pd.DataFrame (n_ref, 978), index = sample IDs
    """
    ds = config.DATASETS[dataset_name]
    path = ds.get("reference_tsv")
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.T


# ─────────────────────────────────────────────────────────────────────────────
# Loading NoisyDiffusion repo data (for training / ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def load_splits_yaml(dataset_name):
    """Load the splits YAML → dict mapping split name to train/test sample IDs."""
    ds = config.DATASETS[dataset_name]
    with open(ds["splits_yaml"]) as f:
        return yaml.safe_load(f)["splits"]


def load_nd_synthetic(dataset_name, split_no):
    """Load NoisyDiffusion repo's labeled synthetic data for a split.

    Returns:
        X : np.ndarray (n_syn, 978)
        y_str : np.ndarray (n_syn,) string labels
    """
    ds = config.DATASETS[dataset_name]
    d = ds["nd_synthetic_dir"]
    X = pd.read_csv(os.path.join(d, f"synthetic_data_split_{split_no}.csv")).values.astype(np.float32)
    y = pd.read_csv(os.path.join(d, f"synthetic_labels_split_{split_no}.csv")).values.ravel()
    return X, y


def generate_custom_splits(dataset_name, n_splits=None, ratio=None, seed=None, force=False):
    """Generate random train/test splits of the real data and save to JSON.

    Each split randomly assigns `ratio` fraction of samples as members (train)
    and the remainder as non-members (test).

    Idempotent: skips if the splits file already exists (unless force=True).
    """
    n_splits = n_splits or config.NUM_CUSTOM_SPLITS
    ratio = ratio or config.CUSTOM_SPLIT_RATIO
    seed = seed or config.SEED

    out_dir = os.path.join(config.CUSTOM_SPLITS_DIR, dataset_name)
    out_path = os.path.join(out_dir, "splits.json")

    if not force and os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) >= n_splits:
            print(f"  Custom splits already exist ({len(existing)} splits): {out_path} (skipping)")
            return out_path
        print(f"  Existing splits.json has {len(existing)} splits but need {n_splits} — regenerating")

    X_real, _ = load_real_data(dataset_name)
    all_ids = list(X_real.index)
    n_samples = len(all_ids)
    n_train = int(n_samples * ratio)

    splits = {}
    for i in range(1, n_splits + 1):
        rng = np.random.RandomState(seed + i)
        perm = rng.permutation(n_samples)
        train_idx = sorted(perm[:n_train].tolist())
        test_idx = sorted(perm[n_train:].tolist())
        splits[f"split_{i}"] = {
            "train_ids": [all_ids[j] for j in train_idx],
            "test_ids": [all_ids[j] for j in test_idx],
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Generated {n_splits} custom splits → {out_path}")
    return out_path


def load_custom_splits(dataset_name):
    """Load saved custom splits from JSON."""
    path = os.path.join(config.CUSTOM_SPLITS_DIR, dataset_name, "splits.json")
    with open(path) as f:
        return json.load(f)


def load_real_split(dataset_name, split_no):
    """Load the real-data training subset for a given CV split.

    Dispatches on config.SPLIT_MODE:
      - "custom": uses saved custom splits (train_ids)
      - "noisydiffusion": uses splits YAML (complement of test_index)

    Returns:
        X_train : np.ndarray, shape (n_train, 978) – member samples for this split
    """
    X_real, _ = load_real_data(dataset_name)

    if config.SPLIT_MODE == "custom":
        splits = load_custom_splits(dataset_name)
        train_ids = set(splits[f"split_{split_no}"]["train_ids"])
        train_mask = np.array([sid in train_ids for sid in X_real.index])
    else:
        splits = load_splits_yaml(dataset_name)
        test_ids = set(splits[f"split_{split_no}"]["test_index"])
        train_mask = np.array([sid not in test_ids for sid in X_real.index])

    return X_real.values[train_mask].astype(np.float32)


def get_membership_labels(dataset_name, split_no):
    """Determine ground-truth membership for one CV split.

    Members (1) = samples in the training set for this split.
    Non-members (0) = samples in the test set for this split.

    Dispatches on config.SPLIT_MODE:
      - "custom": uses saved custom splits (test_ids)
      - "noisydiffusion": uses splits YAML (test_index)

    Returns:
        sample_ids : list[str]  – all sample IDs in the real data
        y_member : np.ndarray (n_samples,) – 0/1 membership labels
    """
    X_real, _ = load_real_data(dataset_name)
    all_ids = list(X_real.index)

    if config.SPLIT_MODE == "custom":
        splits = load_custom_splits(dataset_name)
        test_ids = set(splits[f"split_{split_no}"]["test_ids"])
    else:
        splits = load_splits_yaml(dataset_name)
        test_ids = set(splits[f"split_{split_no}"]["test_index"])

    y_member = np.array([0 if sid in test_ids else 1 for sid in all_ids], dtype=np.int64)
    return all_ids, y_member


def get_nd_membership_labels(dataset_name, split_no):
    """Get ground-truth membership labels from NoisyDiffusion splits YAML.

    Always uses NoisyDiffusion splits regardless of SPLIT_MODE.
    Used by synthetic validation step (which evaluates ND synthetic models).
    """
    X_real, _ = load_real_data(dataset_name)
    all_ids = list(X_real.index)
    splits = load_splits_yaml(dataset_name)
    test_ids = set(splits[f"split_{split_no}"]["test_index"])
    y_member = np.array([0 if sid in test_ids else 1 for sid in all_ids], dtype=np.int64)
    return all_ids, y_member


# ─────────────────────────────────────────────────────────────────────────────
# Label encoding / inference
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y_str, label_list):
    """Map string labels to integer indices."""
    mapping = {c: i for i, c in enumerate(label_list)}
    return np.array([mapping[l] for l in y_str], dtype=np.int64)


def infer_labels_knn(X_unlabeled, X_labeled, y_labeled_int, k=5):
    """Predict integer labels for unlabeled samples using KNN on labeled data.

    Both X arrays should be in the same feature space (raw gene expression).
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, y_labeled_int)
    return knn.predict(X_unlabeled)


def build_label_predictor(dataset_name, split_no=1):
    """Train a KNN label predictor on the NoisyDiffusion repo's labeled synthetic data.

    Returns a function: predict(X_raw) → y_int
    """
    ds = config.DATASETS[dataset_name]
    X_syn, y_syn_str = load_nd_synthetic(dataset_name, split_no)
    y_syn_int = encode_labels(y_syn_str, ds["label_list"])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_syn, y_syn_int)
    return knn.predict


# ─────────────────────────────────────────────────────────────────────────────
# Scaler utilities
# ─────────────────────────────────────────────────────────────────────────────

def fit_quantile_scaler(X):
    """Fit QuantileTransformer on X, return (scaler, X_transformed)."""
    scaler = QuantileTransformer(output_distribution="normal")
    X_t = scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
    return scaler, X_t

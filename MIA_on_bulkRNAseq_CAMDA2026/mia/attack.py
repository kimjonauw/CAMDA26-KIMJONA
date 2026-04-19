"""End-to-end MIA pipeline orchestration.

Pipeline overview (principled shadow-model approach):
  1. Train 5 shadow models on real data subsets (one per YAML split).
     Each split's ~871 training samples are known members for that model.
  2. Extract loss-trajectory features for all real samples using each shadow model.
     Scaler is fitted on the same real data subset used for training that shadow.
  3. Train MLP classifier on features from splits 1-3, validate on splits 4-5.
     Labels are known ground truth (in/out of each shadow's training set).
  4. Challenge inference: train a "target proxy" on challenge synthetic data,
     extract features for all real samples, apply MLP.
"""

import os
import numpy as np
import pandas as pd
import torch

from . import config
from .data_utils import (
    load_real_data,
    load_real_split,
    load_challenge_synthetic,
    load_nd_synthetic,
    get_membership_labels,
    get_nd_membership_labels,
    generate_custom_splits,
    fit_quantile_scaler,
)
from .shadow_model import (
    train_shadow_model,
    train_target_proxy,
    load_shadow_model,
    load_target_proxy,
)
from .loss_features import extract_loss_features, prepare_features
from .classifier import train_classifier, load_classifier, _tpr_at_fpr


def _force(stage):
    """Return True if the given stage should be forced to re-run."""
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


def _num_shadow_splits():
    """Return the number of shadow splits based on SPLIT_MODE."""
    if config.SPLIT_MODE == "custom":
        return config.NUM_CUSTOM_SPLITS
    return config.NUM_SPLITS


def _default_train_val_splits():
    """Compute default 70/30 train/val split indices."""
    N = _num_shadow_splits()
    n_train = max(1, int(N * 0.7))
    train_splits = list(range(1, n_train + 1))
    val_splits = list(range(n_train + 1, N + 1))
    return train_splits, val_splits


# ── Step 0 (conditional) ────────────────────────────────────────────────────

def step_generate_splits(dataset_name):
    """Generate custom random splits when SPLIT_MODE == 'custom'."""
    generate_custom_splits(dataset_name)


# ── Step 1 ───────────────────────────────────────────────────────────────────

def step_train_shadows(dataset_name, splits=None, device=None):
    """Train shadow models on real data subsets (one per split).  Skips if checkpoint exists."""
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    force = _force("shadows")
    for s in splits:
        save_dir = os.path.join(config.SHADOW_MODEL_DIR, dataset_name)
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  Shadow split {s} exists: {save_path} (skipping)")
            continue
        print(f"\n{'='*60}")
        print(f"Training shadow model: {dataset_name} split {s}")
        print(f"{'='*60}")
        X_train = load_real_split(dataset_name, s)
        train_shadow_model(X_train, save_path, split_no=s, device=device)


# ── Step 2 ───────────────────────────────────────────────────────────────────

def step_extract_features(dataset_name, splits=None, device=None):
    """Extract loss features for all real samples, using each split's shadow model.

    For each split, we:
      - Load the shadow model
      - Fit scaler on the same real data subset used for training that shadow
      - Scale all real data with that scaler
      - Extract loss-trajectory features (dummy label, unconditional)
      - Save alongside ground-truth membership labels from the splits YAML
    """
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    os.makedirs(os.path.join(config.FEATURES_DIR, dataset_name), exist_ok=True)

    # Load real data once
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Dummy labels for all real samples
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(config.FEATURES_DIR, dataset_name, f"features_split_{s}.npz")
        if not force and os.path.exists(out_path):
            print(f"  Features split {s} exist: {out_path} (skipping)")
            continue
        print(f"\n{'='*60}")
        print(f"Extracting features: {dataset_name} split {s}")
        print(f"{'='*60}")

        model, diff_trainer = load_shadow_model(dataset_name, s, device=device)

        # Fit scaler on the same real data subset used for training this shadow
        X_train_split = load_real_split(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_train_split)

        # Scale all real data with this scaler
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        # Extract features
        features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)

        # Ground-truth membership
        _, y_member = get_membership_labels(dataset_name, s)

        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}")


# ── Step 3 ───────────────────────────────────────────────────────────────────

def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    """Train the MLP on extracted features from some splits, validate on others."""
    if train_splits is None or val_splits is None:
        default_train, default_val = _default_train_val_splits()
        train_splits = train_splits or default_train
        val_splits = val_splits or default_val
    device = device or config.DEVICE
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
            Xs.append(prepare_features(d["features"]))
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    print(f"\nMLP train: {len(X_train)} samples (dim={X_train.shape[1]}) "
          f"(members={y_train.sum()}, non-members={len(y_train)-y_train.sum()})")
    print(f"MLP val:   {len(X_val)} samples "
          f"(members={y_val.sum()}, non-members={len(y_val)-y_val.sum()})")

    model, history = train_classifier(X_train, y_train, X_val, y_val, device=device)
    return model, history


# ── Synthetic validation ─────────────────────────────────────────────────────

def step_validate_synthetic(dataset_name, device=None):
    """Train models on ND synthetic data and evaluate the MLP — measures domain gap.

    For each of 5 NoisyDiffusion splits:
      1. Train a diffusion model on the ND synthetic data for that split
      2. Fit scaler on that synthetic data
      3. Scale all real data, extract loss features
      4. Load trained MLP, predict membership scores
      5. Compare to ND ground-truth labels
    """
    device = device or config.DEVICE

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    clf = load_classifier()
    clf.eval()

    os.makedirs(os.path.join(config.SYNTH_VAL_MODEL_DIR, dataset_name), exist_ok=True)
    os.makedirs(os.path.join(config.SYNTH_VAL_FEATURES_DIR, dataset_name), exist_ok=True)

    force_sv = _force("synth_val")
    for s in range(1, config.NUM_SPLITS + 1):
        print(f"\n  --- Synthetic validation: split {s} ---")

        feat_path = os.path.join(
            config.SYNTH_VAL_FEATURES_DIR, dataset_name, f"features_synth_val_split_{s}.npz"
        )

        if not force_sv and os.path.exists(feat_path):
            print(f"    Features exist: {feat_path} (loading)")
            d = np.load(feat_path)
            features, y_member_nd = d["features"], d["y_member"]
        else:
            # Load ND synthetic data (ignore labels — unconditional model)
            X_syn, _ = load_nd_synthetic(dataset_name, s)

            # Train diffusion model on synthetic data
            save_path = os.path.join(
                config.SYNTH_VAL_MODEL_DIR, dataset_name, f"synth_val_split_{s}.pt"
            )
            model, diff_trainer, scaler = train_shadow_model(
                X_syn, save_path, split_no=100 + s, device=device
            )

            # Scale all real data with synth-fitted scaler
            X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

            # Extract features
            features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)

            _, y_member_nd = get_nd_membership_labels(dataset_name, s)
            np.savez(feat_path, features=features, y_member=y_member_nd)

        # Predict with MLP (always re-run — classifier may have changed)
        features_prep = prepare_features(features)
        with torch.no_grad():
            scores = clf(torch.tensor(features_prep, dtype=torch.float32)).squeeze(1).numpy()

        tpr = _tpr_at_fpr(y_member_nd, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member_nd).mean()
        print(f"  Synth-val split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member_nd.sum()}, non-members={len(y_member_nd)-y_member_nd.sum()})")


# ── Challenge inference ──────────────────────────────────────────────────────

def step_predict_challenge(dataset_name, device=None):
    """Produce membership predictions for the challenge's real data.

    Trains (or loads) a target proxy model on challenge synthetic data,
    fits scaler on challenge synthetic data, extracts features, applies MLP.
    """
    device = device or config.DEVICE

    # Load real data
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Train or load target proxy
    proxy_ckpt = os.path.join(config.SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        print("  Loading existing target proxy...")
        model, diff_trainer = load_target_proxy(dataset_name, device=device)
        # Reconstruct scaler by fitting on challenge synthetic data
        X_syn = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        print("  Training target proxy on challenge synthetic data...")
        model, diff_trainer, scaler = train_target_proxy(dataset_name, device=device)

    # Dummy labels + scale
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)
    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

    # Extract features
    features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)
    features = prepare_features(features)

    # Classify
    clf = load_classifier()
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    # Save predictions
    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")

    return sample_ids, scores


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_name, device=None):
    """Run the complete MIA pipeline for one dataset."""
    device = device or config.DEVICE
    N = _num_shadow_splits()

    # ── Step 0: Generate custom splits (if applicable) ───────────────────
    if config.SPLIT_MODE == "custom":
        print("\n" + "=" * 70)
        print(f"STEP 0: Generating {N} custom splits ({dataset_name})")
        print("=" * 70)
        step_generate_splits(dataset_name)

    print(f"\n  Split mode: {config.SPLIT_MODE}, {N} splits")

    # ── Step 1: Train shadow models ──────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 1: Training shadow models on real data subsets ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    # ── Step 2: Extract loss features ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting loss features ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    # ── Step 3: Train MLP classifier ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    train_splits, val_splits = _default_train_val_splits()
    print(f"  Train splits: {train_splits}, Val splits: {val_splits}")
    model, history = step_train_classifier(
        dataset_name, train_splits=train_splits, val_splits=val_splits, device=device
    )

    # ── Step 4: Per-split evaluation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)

    for s in range(1, N + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        features, y_member = prepare_features(d["features"]), d["y_member"]

        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

        tpr = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member).mean()
        print(f"  Split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member.sum()}, non-members={len(y_member)-y_member.sum()})")

    # ── Step 5: Synthetic validation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 5: Synthetic validation — domain gap measurement ({dataset_name})")
    print("=" * 70)
    step_validate_synthetic(dataset_name, device=device)

    # ── Step 6: Challenge predictions ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 6: Challenge predictions ({dataset_name})")
    print("=" * 70)
    step_predict_challenge(dataset_name, device=device)

    return model, history


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="BRCA")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--profile", choices=list(config.PROFILES.keys()), default=None,
                        help="Named config profile (default: use current config values)")
    parser.add_argument("--force", default="",
                        help="Force re-run of stages (comma-separated: shadows,features,classifier,"
                             "synth_val,challenge) or 'all'")
    args = parser.parse_args()
    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_profile(args.profile)
    print(f"Active profile: {config.ACTIVE_PROFILE}  "
          f"(FEATURE_MODE={config.FEATURE_MODE}, MLP_INPUT_DIM={config.MLP_INPUT_DIM}, "
          f"MLP_DROPOUT={config.MLP_DROPOUT}, MLP_WEIGHT_DECAY={config.MLP_WEIGHT_DECAY})")
    run_full_pipeline(args.dataset, device=args.device)

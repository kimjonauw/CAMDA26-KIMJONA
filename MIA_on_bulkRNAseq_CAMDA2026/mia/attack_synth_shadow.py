"""Synth-shadow MIA pipeline with superset slicing + Optuna joint search.

KEY FIX (2026-04-10):
  step_evaluate_classifiers previously re-scored every split using the FINAL
  model (trained on ALL 5 splits), causing LGBM/CAT to show AUC=1.000 on
  in-sample data.  Now Step 3 caches per-fold LOSO scores and Step 4 reuses
  them directly — so every reported per-split number is a genuine held-out
  estimate, and ensemble blend weights are computed on the same honest scores.

KEY FIX (2026-04-20) — Optuna feature-selection leak (hard seal):
  Optuna previously ran on ALL 5 splits pooled, letting it see held-out eval
  splits when choosing the T-subset (feature selection) and noise budget.

  Fix: splits are partitioned into two non-overlapping zones:
    OPTUNA_ZONE  (splits 1-3, 0-based idx 0-2):
        Optuna sees ONLY these splits. T-subset and hparams are tuned here.
    EVAL_ZONE    (splits 4-5, 0-based idx 3-4):
        _loso_cv_tpr iterates val_s over EVAL_ZONE indices ONLY.
        But the training pool inside each fold is ALL other splits (1-4 when
        val=5, 1-3+5 when val=4) — so no training starvation.

  The final model (Step 5, challenge inference) still trains on all splits.

Architecture:
  - Step 2: extract FULL superset grid (n_samples, N_T_SUPERSET * N_NOISE) once
  - Optuna (trees only): T-subset + hparams tuned on OPTUNA_ZONE (S1-S3) only
  - MLP: fixed signal params, no Optuna
  - LOSO-CV: val_s iterates over EVAL_ZONE; training uses all other splits
  - Final model: trained on ALL 5 splits (for challenge inference only)
  - Evaluation / ensemble: uses LOSO-cached scores from EVAL_ZONE
"""

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from . import config
from .data_utils import (
    load_real_data,
    load_nd_synthetic,
    load_challenge_synthetic,
    get_nd_membership_labels,
    fit_quantile_scaler,
)
from .shadow_model import (
    train_shadow_model,
    build_model,
    build_diffusion_trainer,
    train_target_proxy,
    load_target_proxy,
    load_shadow_scaler, 
    _scaler_path,        
)
from .loss_features import (
    extract_loss_features,
    slice_raw_features,
    summarize_features,
    prepare_features,
)
from .classifier import (
    get_classifier,
    _tpr_at_fpr,
    tpr_at_fpr_multi,
    compute_softmax_weights,
    MembershipMLP,
    train_classifier,
    load_classifier,
)

# ── MLP fixed signal + architecture params (no Optuna) ───────────────────────
_NO_OPTUNA_CLASSIFIERS = {"mlp"}

_MLP_T_LIST     = [1, 2, 5, 10, 20, 50, 100, 200]
_MLP_N_NOISE    = 100
_MLP_HIDDEN_DIM = 64
_MLP_EPOCHS     = 2000

# Legacy aliases kept for report / JSON back-compat
TRAIN_SPLITS = list(range(1, config.NUM_SPLITS + 1))
VAL_SPLITS   = list(range(1, config.NUM_SPLITS + 1))


def _force(stage):
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


# ── Directory helpers ─────────────────────────────────────────────────────────
def _shadow_model_dir(dn):  return os.path.join(config.SYNTH_SHADOW_MODEL_DIR,      dn)
def _features_dir(dn):      return os.path.join(config.SYNTH_SHADOW_FEATURES_DIR,   dn)
def _classifier_dir(cn):    return os.path.join(config.SYNTH_SHADOW_CLASSIFIER_DIR, cn)
def _optuna_dir(dn, cn):    return os.path.join(config.SYNTH_SHADOW_OPTUNA_DIR,     dn, cn)


# ── Timestep grouping ─────────────────────────────────────────────────────────
#   T_SUPERSET = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750]
#   idx           0  1  2   3   4   5   6   7   8    9   10   11   12   13   14
T_GROUPS = {
    "very_early": [0, 1, 2],        # t=1,2,5     — near-zero noise
    "early":      [3, 4, 5, 6],     # t=10,20,30,40
    "mid":        [7, 8, 9],        # t=50,75,100
    "late":       [10, 11, 12],     # t=150,200,300
    "very_late":  [13, 14],         # t=500,750    — near-Gaussian
}
T_GROUP_NAMES = list(T_GROUPS.keys())


def _build_t_indices(trial):
    selected = []
    for name in T_GROUP_NAMES:
        if trial.suggest_categorical(f"use_{name}", [True, False]):
            selected.extend(T_GROUPS[name])
    if len(selected) < 3:
        fallback = T_GROUPS["early"] + T_GROUPS["mid"]
        selected = sorted(set(selected) | set(fallback))
    return sorted(set(selected))


def _suggest_xgb(trial):
    return {
        "max_depth":        trial.suggest_int("max_depth",          2, 6),
        "learning_rate":    trial.suggest_float("learning_rate",    0.005, 0.1,  log=True),
        "subsample":        trial.suggest_float("subsample",        0.4, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight",   5, 50),
        "gamma":            trial.suggest_float("gamma",            0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",        0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda",       0.5, 10.0),
        "n_estimators":     2000,
        "tree_method":      "hist",
    }


def _suggest_rf(trial):
    return {
        "n_estimators":     trial.suggest_categorical("n_estimators",   [200, 500, 1000]),
        "max_depth":        trial.suggest_int("max_depth",              4, 20),
        "max_features":     trial.suggest_categorical("max_features",   ["sqrt", "log2", 0.2, 0.3]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",       5, 50),
        "max_samples":      trial.suggest_float("max_samples",          0.4, 0.9),
    }


def _suggest_lgbm(trial):
    return {
        "num_leaves":        trial.suggest_int("num_leaves",         16, 128),
        "learning_rate":     trial.suggest_float("learning_rate",    0.005, 0.05, log=True),
        "subsample":         trial.suggest_float("subsample",        0.6, 0.9),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "min_child_samples": trial.suggest_int("min_child_samples",  5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha",        0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda",       0.0, 2.0),
        "n_estimators":      3000,
        "min_split_gain":    trial.suggest_float("min_split_gain",   0.0, 1.0),
        "boosting_type":     "gbdt",
        "verbose":           -1,
    }


def _suggest_cat(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
    params = {
        "bootstrap_type": bootstrap_type,
        "iterations":     trial.suggest_int("iterations",    100, 1000),
        "depth":          trial.suggest_int("depth",         4, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    return params


_SUGGEST = {
    "xgb": _suggest_xgb, "rf": _suggest_rf,
    "lgbm": _suggest_lgbm, "cat": _suggest_cat,
}


# ── Optuna objective — row-level KFold on OPTUNA_ZONE only ───────────────────
def _make_objective(clf_name, raw_opt, y_opt, n_folds):
    """Optuna objective — row-level KFold on OPTUNA_ZONE splits ONLY.
    EVAL_ZONE splits (S4, S5) are never passed here, so T-subset selection
    cannot be informed by held-out evaluation data.
    """
    entry   = get_classifier(clf_name)
    suggest = _SUGGEST[clf_name]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    def objective(trial):
        t_indices    = _build_t_indices(trial)
        noise_budget = trial.suggest_int("n_noise", 150, config.N_NOISE, step=50)
        n_t          = len(t_indices)

        X_sliced = slice_raw_features(raw_opt, t_indices, noise_budget)
        X_feat   = summarize_features(X_sliced, noise_budget, n_t)
        hparams  = suggest(trial)

        fold_tprs = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_feat, y_opt)):
            X_tr, y_tr = X_feat[tr_idx], y_opt[tr_idx]
            X_vl, y_vl = X_feat[vl_idx], y_opt[vl_idx]
            tmp_dir = f"/tmp/optuna_{clf_name}_t{trial.number}_f{fold_idx}"
            try:
                clf, _  = entry.train(X_tr, y_tr, X_vl, y_vl, tmp_dir, hparams=hparams)
                scores  = entry.predict(clf, X_vl)
                fold_tprs.append(_tpr_at_fpr(y_vl.astype(int), scores))
            except Exception as e:
                print(f"  [Optuna/{clf_name}] trial={trial.number} fold={fold_idx} error: {e}")
                fold_tprs.append(0.0)

        return float(np.mean(fold_tprs))

    return objective


def run_optuna(clf_name, raw_opt, y_opt, dataset_name):
    """Run Optuna on OPTUNA_ZONE splits (S1-S3) only. Returns best params dict."""
    out_dir    = _optuna_dir(dataset_name, clf_name)
    os.makedirs(out_dir, exist_ok=True)

    best_path  = os.path.join(out_dir, "best_params.json")
    db_path    = os.path.join(out_dir, "study.db")
    storage    = config.OPTUNA_STORAGE or f"sqlite:///{db_path}"
    study_name = f"{dataset_name}_{clf_name}_{config.ACTIVE_PROFILE}"

    if _force("classifier"):
        if os.path.exists(best_path):
            os.remove(best_path)
            print(f"  [Optuna/{clf_name.upper()}] FORCE: wiped cached best_params.json")
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"  [Optuna/{clf_name.upper()}] FORCE: wiped study '{study_name}'")
        except KeyError:
            pass
        if not config.OPTUNA_STORAGE and os.path.exists(db_path):
            os.remove(db_path)

    if not _force("classifier") and os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
        print(f"  [Optuna/{clf_name.upper()}] Loaded cached best params: {best}")
        return best

    study = optuna.create_study(
        study_name     = study_name,
        direction      = "maximize",
        storage        = storage,
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=config.SEED),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0),
    )

    n_trials = config.OPTUNA_N_TRIALS
    n_folds  = config.OPTUNA_CV_FOLDS
    print(f"\n  [Optuna/{clf_name.upper()}] {n_trials} trials × {n_folds}-fold CV  "
          f"on OPTUNA_ZONE splits only (leak-free) ...")

    objective = _make_objective(clf_name, raw_opt, y_opt, n_folds)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=config.OPTUNA_TIMEOUT,
        n_jobs=1,
        show_progress_bar=False,
    )

    best     = study.best_params
    best_val = study.best_value
    print(f"  [Optuna/{clf_name.upper()}] Best mean TPR@10%FPR = {best_val:.4f}")
    print(f"  [Optuna/{clf_name.upper()}] Best params: {best}")

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    return best


def _params_to_t_indices_and_noise(best_params):
    t_indices = []
    for name in T_GROUP_NAMES:
        if best_params.get(f"use_{name}", False):
            t_indices.extend(T_GROUPS[name])
    if len(t_indices) < 3:
        fallback = T_GROUPS["early"] + T_GROUPS["mid"]
        t_indices = sorted(set(t_indices) | set(fallback))
    else:
        t_indices = sorted(set(t_indices))
    noise_budget = int(best_params.get("n_noise", config.N_NOISE))
    return t_indices, noise_budget


def _extract_clf_hparams(best_params, clf_name):
    keep = {}
    for k, v in best_params.items():
        if k.startswith("use_") or k == "n_noise":
            continue
        keep[k] = v
    return keep


# ── Step 1 ────────────────────────────────────────────────────────────────────
def step_train_shadows(dataset_name, splits=None, device=None):
    splits    = splits or list(range(1, config.NUM_SPLITS + 1))
    device    = device or config.DEVICE
    save_dir  = _shadow_model_dir(dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    force = _force("shadows")
    for s in splits:
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  Synth-shadow split {s} exists: {save_path} (skipping)")
            continue
        print(f"\n{'='*60}\nTraining synth-shadow: {dataset_name} split {s}\n{'='*60}")
        X_syn, y_str = load_nd_synthetic(dataset_name, s)
        train_shadow_model(
            X_train=X_syn,
            save_path=save_path,
            split_no=s,
            device=device,
            y_str=y_str,
            dataset_name=dataset_name,
        )

# ── Step 2 ────────────────────────────────────────────────────────────────────
def step_extract_features(dataset_name, splits=None, device=None):
    """Extract FULL superset grid once — shape (n_samples, N_T_SUPERSET * N_NOISE).

    FIX (2026-04-20): load the exact QuantileTransformer that was fitted on the
    post-SMOTE synthetic data during Step 1, rather than re-fitting a new one on
    the raw (pre-SMOTE) X_syn.  The shadow model's weights are calibrated to the
    post-SMOTE quantile boundaries; applying a different scaler shifts X_real
    out-of-distribution relative to those weights, degrading loss trajectory
    discrimination between members and non-members.
    """
    splits    = splits or list(range(1, config.NUM_SPLITS + 1))
    device    = device or config.DEVICE
    feat_dir  = _features_dir(dataset_name)
    os.makedirs(feat_dir, exist_ok=True)
    model_dir = _shadow_model_dir(dataset_name)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np    = X_real_df.values.astype(np.float32)
    sample_ids   = list(X_real_df.index)
    y_int        = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(feat_dir, f"features_split_{s}.npz")

        needs_rerun = force
        if not needs_rerun and os.path.exists(out_path):
            meta          = np.load(out_path)
            stored_shape  = meta["features"].shape
            expected_cols = len(config.T_SUPERSET) * config.N_NOISE
            if stored_shape[1] != expected_cols:
                print(f"  Features split {s}: shape mismatch "
                      f"(stored={stored_shape[1]}, expected={expected_cols}) — re-extracting")
                needs_rerun = True
            else:
                print(f"  Features split {s} exist: {out_path} (skipping)")
                continue

        print(f"\n{'='*60}\nExtracting superset features: {dataset_name} split {s}\n{'='*60}")
        print(f"  T_SUPERSET={config.T_SUPERSET}, N_NOISE={config.N_NOISE}")

        model = build_model(config.UNCONDITIONAL_NUM_CLASSES, device)
        ckpt  = os.path.join(model_dir, f"shadow_split_{s}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        diff_trainer = build_diffusion_trainer(device)

        # FIX: load the scaler that was fitted on post-SMOTE X_syn during Step 1,
        # not a freshly fitted one on raw X_syn.  This guarantees the quantile
        # boundaries match exactly what the shadow model was trained on.
        scaler = load_shadow_scaler(ckpt)
        print(f"  Loaded shadow scaler from {_scaler_path(ckpt)}")

        X_scaled  = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        features    = extract_loss_features(
            model, diff_trainer, X_scaled, y_int,
            t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device
        )
        _, y_member = get_nd_membership_labels(dataset_name, s)

        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}  "
              f"(expected cols={len(config.T_SUPERSET) * config.N_NOISE})")
# ── LOSO-CV — caches per-fold scores for honest evaluation in Step 4 ─────────
def _loso_cv_tpr(clf_name, entry, raw_splits, y_splits,
                 t_indices, noise_budget, clf_hparams,
                 eval_indices=None):
    """Leave-one-split-out CV.

    Parameters
    ----------
    eval_indices : list[int] or None
        0-based indices into raw_splits that are used as the validation set.
        Training pool for each fold = ALL other splits (not just other eval
        splits).  If None, defaults to range(len(raw_splits)) — original
        behaviour where every split is val exactly once.

    Returns
    -------
    mean_tpr      : float  — mean TPR@10%FPR across eval folds
    mean_auc      : float  — mean AUC across eval folds
    fold_tprs     : list   — per-fold TPR@10%FPR
    fold_scores   : dict   — {val_split_idx (0-based): np.ndarray of scores}
    fold_y        : dict   — {val_split_idx (0-based): np.ndarray of labels}
    fold_aucs     : list   — per-fold AUC
    """
    n_splits    = len(raw_splits)
    if eval_indices is None:
        eval_indices = list(range(n_splits))
    fold_tprs   = []
    fold_aucs   = []
    fold_scores = {}   # keyed by 0-based split index (e.g. 3, 4 for EVAL_ZONE)
    fold_y      = {}

    for val_s in eval_indices:  # iterate ONLY over eval folds
        train_idx = [i for i in range(n_splits) if i != val_s]

        raw_tr = np.concatenate([raw_splits[i] for i in train_idx])
        y_tr   = np.concatenate([y_splits[i]   for i in train_idx])
        raw_vl = raw_splits[val_s]
        y_vl   = y_splits[val_s]

        X_tr = summarize_features(
            slice_raw_features(raw_tr, t_indices, noise_budget),
            noise_budget, len(t_indices)
        )
        X_vl = summarize_features(
            slice_raw_features(raw_vl, t_indices, noise_budget),
            noise_budget, len(t_indices)
        )

        try:
            clf, _ = entry.train(
                X_tr, y_tr, X_vl, y_vl,
                f"/tmp/loso_{clf_name}_fold{val_s}",
                hparams=clf_hparams
            )
            scores = entry.predict(clf, X_vl)
            tpr    = _tpr_at_fpr(y_vl.astype(int), scores)
            auc    = (roc_auc_score(y_vl.astype(int), scores)
                      if len(np.unique(y_vl)) > 1 else 0.5)

            print(f"    [LOSO S{val_s+1} held out]  "
                  f"train=S{[i+1 for i in train_idx]}  "
                  f"TPR@10%={tpr:.4f}  AUC={auc:.4f}  "
                  f"(members={int(y_vl.sum())}, non={len(y_vl)-int(y_vl.sum())})")

            fold_tprs.append(tpr)
            fold_aucs.append(auc)
            fold_scores[val_s] = scores
            fold_y[val_s]      = y_vl

        except Exception as e:
            print(f"    [LOSO S{val_s+1}] error: {e}")
            fold_tprs.append(0.0)
            fold_aucs.append(0.5)
            fold_scores[val_s] = np.zeros(len(y_splits[val_s]))
            fold_y[val_s]      = y_splits[val_s]

    mean_tpr = float(np.mean(fold_tprs))
    mean_auc = float(np.mean(fold_aucs))
    print(f"  [LOSO] Mean TPR@10%FPR={mean_tpr:.4f}  "
          f"Mean AUC={mean_auc:.4f}  "
          f"std={float(np.std(fold_tprs)):.4f}  "
          f"per-fold={[f'{t:.3f}' for t in fold_tprs]}")

    return mean_tpr, mean_auc, fold_tprs, fold_scores, fold_y, fold_aucs


# ── Step 3 ────────────────────────────────────────────────────────────────────
def step_train_classifiers(dataset_name, device=None):
    """Train classifiers with leak-free Optuna + EVAL_ZONE LOSO estimate.

    Flow:
      1. Optuna (tree models only) — row-level KFold on OPTUNA_ZONE only.
         Finds best hparams + signal params (T-subset, n_noise).  EVAL_ZONE is
         completely sealed off.
      2. LOSO-CV — val restricted to EVAL_ZONE only, but each fold trains on
         ALL non-val splits.  Per-fold scores are cached for Step 4.
      3. Final model — trains on ALL 5 splits with best hparams.
         Used ONLY for challenge inference (Step 5).

    Returns
    -------
    dict  clf_name →
        (clf, history, t_indices, noise_budget,
         val_tpr, val_auc, selected_ts, loso_fold_tprs,
         loso_fold_scores, loso_fold_y, loso_fold_aucs)
    """
    device   = device or config.DEVICE
    feat_dir = _features_dir(dataset_name)
    n_splits = config.NUM_SPLITS

    raw_splits, y_splits = [], []
    for s in range(1, n_splits + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        raw_splits.append(d["features"])
        y_splits.append(d["y_member"])

    raw_all = np.concatenate(raw_splits)
    y_all   = np.concatenate(y_splits)

    # ── Zone split: Optuna must never see EVAL_ZONE data ─────────────────────
    # OPTUNA_ZONE  = splits 1..ceil(n*0.6)  — T-subset / hparam search only
    # EVAL_ZONE    = splits ceil(n*0.6)+1..n — val targets for honest LOSO
    #
    # IMPORTANT: _loso_cv_tpr is still passed the FULL raw_splits list so that
    # each eval fold trains on all non-val splits (e.g. S1-S4 when val=S5).
    # eval_indices kwarg restricts WHICH folds are used as validation.
    import math
    n_opt_splits  = math.ceil(n_splits * 0.6)          # 3 of 5 by default
    opt_indices   = list(range(n_opt_splits))           # 0-based: [0, 1, 2]
    eval_indices  = list(range(n_opt_splits, n_splits)) # 0-based: [3, 4]

    raw_opt = np.concatenate([raw_splits[i] for i in opt_indices])
    y_opt   = np.concatenate([y_splits[i]   for i in opt_indices])

    print(f"\nLoaded {n_splits} splits:")
    for i, (r, y) in enumerate(zip(raw_splits, y_splits)):
        zone = "OPTUNA_ZONE" if i in opt_indices else "EVAL_ZONE "
        print(f"  S{i+1} [{zone}]: {r.shape}  members={int(y.sum())}  "
              f"non={len(y)-int(y.sum())}  imbalance={int(y.sum())/len(y):.3f}")
    print(f"Pooled (all 5):       {raw_all.shape}  "
          f"members={int(y_all.sum())}  non={len(y_all)-int(y_all.sum())}")
    print(f"Pooled (optuna S1-S{n_opt_splits}): {raw_opt.shape}  "
          f"members={int(y_opt.sum())}  non={len(y_opt)-int(y_opt.sum())}")
    print(f"Eval zone: S{n_opt_splits+1}-S{n_splits}  "
          f"({len(eval_indices)} folds held-out from Optuna)")

    results = {}
    for clf_name in config.ACTIVE_CLASSIFIERS:
        print(f"\n{'='*60}\n  Classifier: {clf_name.upper()}\n{'='*60}")
        entry    = get_classifier(clf_name)
        save_dir = _classifier_dir(clf_name)

        # ── MLP: fixed signal params, no Optuna ──────────────────────────────
        if clf_name in _NO_OPTUNA_CLASSIFIERS:
            t_indices = [
                config.T_SUPERSET.index(t)
                for t in _MLP_T_LIST
                if t in config.T_SUPERSET
            ]
            if not t_indices:
                raise ValueError(
                    f"None of _MLP_T_LIST={_MLP_T_LIST} found in "
                    f"config.T_SUPERSET={config.T_SUPERSET}"
                )
            noise_budget = _MLP_N_NOISE
            clf_hparams  = {
                "hidden_dim":   _MLP_HIDDEN_DIM,
                "epochs":       _MLP_EPOCHS,
                "dropout":      config.MLP_DROPOUT,
                "weight_decay": config.MLP_WEIGHT_DECAY,
                "lr":           config.MLP_LR,
                "batch_size":   config.MLP_BATCH_SIZE,
            }
            selected_ts = [config.T_SUPERSET[i] for i in t_indices]
            print(f"  [MLP] Fixed signal: T={selected_ts}  n_noise={noise_budget}")

        # ── Tree models: Optuna on OPTUNA_ZONE only ───────────────────────────
        elif config.OPTUNA_ENABLED:
            best_params             = run_optuna(clf_name, raw_opt, y_opt, dataset_name)
            t_indices, noise_budget = _params_to_t_indices_and_noise(best_params)
            clf_hparams             = _extract_clf_hparams(best_params, clf_name)
            selected_ts             = [config.T_SUPERSET[i] for i in t_indices]
            print(f"  Best T subset:     {selected_ts}")
            print(f"  Best noise budget: {noise_budget}")
            print(f"  Best clf hparams:  {clf_hparams}")
        else:
            t_indices    = list(range(len(config.T_SUPERSET)))
            noise_budget = config.N_NOISE
            clf_hparams  = None
            selected_ts  = list(config.T_SUPERSET)

        # ── LOSO-CV: val restricted to EVAL_ZONE; trains on all other splits ──
        # Pass FULL raw_splits so each fold trains on 4 splits (not 1).
        # eval_indices kwarg ensures only S4/S5 are ever used as val targets.
        print(f"\n  [LOSO] {len(eval_indices)}-fold LOSO on EVAL_ZONE "
              f"(S{n_opt_splits+1}-S{n_splits}) "
              f"— T-subset tuned on S1-S{n_opt_splits} only ...")
        (val_tpr, val_auc,
         loso_fold_tprs,
         loso_fold_scores,
         loso_fold_y,
         loso_fold_aucs) = _loso_cv_tpr(
            clf_name, entry, raw_splits, y_splits,   # full 5-split list
            t_indices, noise_budget, clf_hparams,
            eval_indices=eval_indices                 # val on S4, S5 only
        )

        # ── Final model: train on ALL 5 splits (challenge inference only) ────
        X_all = summarize_features(
            slice_raw_features(raw_all, t_indices, noise_budget),
            noise_budget, len(t_indices)
        )
        print(f"\n  [FINAL] Training on all {n_splits} splits  "
              f"dim={X_all.shape[1]}  n={len(X_all)}")
        clf, history = entry.train(X_all, y_all, X_all, y_all, save_dir,
                                   hparams=clf_hparams)

        print(f"  [LOSO val] TPR@10%FPR={val_tpr:.4f}  AUC={val_auc:.4f}")

        meta = {
            "t_indices":      t_indices,
            "noise_budget":   noise_budget,
            "clf_hparams":    clf_hparams or {},
            "selected_ts":    selected_ts,
            "loso_fold_tprs": loso_fold_tprs,
            "loso_fold_aucs": loso_fold_aucs,
            "loso_mean_tpr":  val_tpr,
            "loso_mean_auc":  val_auc,
        }
        with open(os.path.join(save_dir, "signal_params.json"), "w") as f:
            json.dump(meta, f, indent=2)

        results[clf_name] = (
            clf, history,
            t_indices, noise_budget,
            val_tpr, val_auc,
            selected_ts, loso_fold_tprs,
            loso_fold_scores,   # {0-based split idx → scores array}
            loso_fold_y,        # {0-based split idx → labels array}
            loso_fold_aucs,     # per-fold AUC list
        )

    return results


# ── Step 4 ────────────────────────────────────────────────────────────────────
def step_evaluate_classifiers(dataset_name, trained_results):
    """Evaluation using LOSO-cached scores from EVAL_ZONE only.

    FIX (2026-04-10): no final-model re-scoring — avoids in-sample AUC=1.000.
    FIX (2026-04-20): loso_fold_scores are keyed by 0-based EVAL_ZONE indices
    (e.g. 3, 4).  This function maps them back to 1-based global split labels
    (4, 5) for display.  All metrics are from classifiers that trained on
    Splits 1-4 and were tested on Split 5 (and vice versa), with the T-subset
    that was tuned exclusively on S1-S3.
    """
    n_splits    = config.NUM_SPLITS
    fpr_targets = [0.01, 0.05, 0.10, 0.20]

    summary    = {clf_name: {} for clf_name in trained_results}
    summary["ensemble"] = {}

    # ── Phase 1: Collect LOSO scores for EVAL_ZONE splits only ─────────────
    # loso_fold_scores / loso_fold_y are keyed by 0-based split index
    # matching eval_indices (e.g. keys 3, 4 for default 5-split setup).
    # global_s = val_s + 1  converts to the 1-based split label used in logs.
    ref_clf          = next(iter(trained_results))
    loso_keys_0based = sorted(trained_results[ref_clf][9].keys())  # e.g. [3, 4]

    split_scores = {}  # {global_s (1-based): {clf_name: scores}}
    split_y      = {}  # {global_s (1-based): y_member array}

    for val_s in loso_keys_0based:
        global_s = val_s + 1
        split_scores[global_s] = {}
        split_y[global_s] = trained_results[ref_clf][9][val_s]  # loso_fold_y

    for clf_name, entry_tuple in trained_results.items():
        (clf, history, t_indices, noise_budget,
         val_tpr, val_auc, selected_ts, loso_fold_tprs,
         loso_fold_scores, loso_fold_y, loso_fold_aucs) = entry_tuple

        for idx, val_s in enumerate(loso_keys_0based):
            global_s = val_s + 1
            scores   = loso_fold_scores[val_s]
            y_member = loso_fold_y[val_s]

            split_scores[global_s][clf_name] = scores

            tpr       = _tpr_at_fpr(y_member.astype(int), scores)
            auc       = loso_fold_aucs[idx]   # list is positional (len=n_eval_folds)
            multi_fpr = tpr_at_fpr_multi(y_member.astype(int), scores)

            print(
                f"  [{clf_name.upper():4s}] Split {global_s} [EVAL_ZONE LOSO held-out]: "
                f"TPR@1%={multi_fpr[0.01]:.4f}  TPR@5%={multi_fpr[0.05]:.4f}  "
                f"TPR@10%={multi_fpr[0.10]:.4f}  TPR@20%={multi_fpr[0.20]:.4f}  "
                f"AUC={auc:.4f}  "
                f"(members={int(y_member.sum())}, "
                f"non-members={len(y_member)-int(y_member.sum())})"
            )
            summary[clf_name][global_s] = {"tpr": tpr, "auc": auc, "multi_fpr": multi_fpr}

    # ── Phase 2: Softmax weights from LOSO mean AUC ──────────────────────────
    loso_aucs = {n: float(trained_results[n][5]) for n in trained_results}

    gate_threshold = config.ENSEMBLE_MIN_AUC_GATE
    n_above_gate   = sum(1 for v in loso_aucs.values() if v >= gate_threshold)
    if n_above_gate == 0:
        warnings.warn(
            f"[ENSEMBLE] All classifiers below LOSO AUC gate={gate_threshold:.2f}. "
            "Falling back to equal weights across all classifiers."
        )
        gate_threshold = 0.0

    softmax_w = compute_softmax_weights(
        loso_aucs,
        temperature=config.ENSEMBLE_TEMPERATURE,
        min_auc_gate=gate_threshold,
    )

    print(f"\n  [ENSEMBLE WEIGHTS]  "
          f"temperature={config.ENSEMBLE_TEMPERATURE}  "
          f"min_auc_gate={config.ENSEMBLE_MIN_AUC_GATE}  "
          f"(LOSO AUC — EVAL_ZONE only)")
    for clf_name, w in softmax_w.items():
        gated = w < 1e-6
        print(f"  [ENSEMBLE] {'⚠ ' if gated else '  '}"
              f"{clf_name.upper():<6}  weight={w:.4f}  "
              f"loso_auc={loso_aucs[clf_name]:.4f}"
              + (f"  (< gate={config.ENSEMBLE_MIN_AUC_GATE})" if gated else ""))

    # ── Phase 3: Ensemble blend on LOSO held-out scores ──────────────────────
    print(f"\n  [ENSEMBLE EVALUATION]  (EVAL_ZONE LOSO held-out scores only)")
    for global_s in sorted(split_y.keys()):
        y_member   = split_y[global_s]
        ens_scores = np.zeros(len(y_member), dtype=np.float64)

        for clf_name, w in softmax_w.items():
            ens_scores += split_scores[global_s][clf_name] * w

        tpr       = _tpr_at_fpr(y_member.astype(int), ens_scores)
        auc       = (roc_auc_score(y_member.astype(int), ens_scores)
                     if len(np.unique(y_member)) > 1 else 0.5)
        multi_fpr = tpr_at_fpr_multi(y_member.astype(int), ens_scores)

        print(
            f"  [ENSM] Split {global_s} [EVAL_ZONE LOSO held-out]: "
            f"TPR@1%={multi_fpr[0.01]:.4f}  TPR@5%={multi_fpr[0.05]:.4f}  "
            f"TPR@10%={multi_fpr[0.10]:.4f}  TPR@20%={multi_fpr[0.20]:.4f}  "
            f"AUC={auc:.4f}"
        )
        summary["ensemble"][global_s] = {"tpr": tpr, "auc": auc, "multi_fpr": multi_fpr}

    # ── Summary table ─────────────────────────────────────────────────────────
    split_cols = "  ".join(
        f"S{s}@10%  AUC" for s in sorted(split_y.keys())
    )
    hdr = (f"  {'CLF':<8}  {'LOSO TPR':>9}  {'LOSO AUC':>9}  "
           f"{split_cols}  {'m@10%':>7}  {'mAUC':>7}  T-subset")
    sep = "─" * max(len(hdr), 120)
    print(f"\n{sep}\n{hdr}\n{sep}")

    eval_global_keys = sorted(split_y.keys())
    for clf_name, splits in summary.items():
        if clf_name == "ensemble":
            val_tpr, val_auc, selected_ts = np.nan, np.nan, "N/A (Blend)"
        else:
            val_tpr, val_auc, selected_ts = (
                trained_results[clf_name][4],
                trained_results[clf_name][5],
                trained_results[clf_name][6],
            )

        split_vals = "  ".join(
            f"{splits[s]['multi_fpr'][0.10]:6.4f}  {splits[s]['auc']:6.4f}"
            for s in eval_global_keys
        )
        tprs = [splits[s]["tpr"] for s in eval_global_keys]
        aucs = [splits[s]["auc"] for s in eval_global_keys]
        vt   = f"{val_tpr:9.4f}" if not np.isnan(val_tpr) else "      ---"
        va   = f"{val_auc:9.4f}" if not np.isnan(val_auc) else "      ---"

        print(f"  {clf_name.upper():<8}  {vt}  {va}  "
              f"{split_vals}  {np.mean(tprs):7.4f}  {np.mean(aucs):7.4f}  {selected_ts}")
    print(sep)

    return summary, softmax_w


# ── Step 5 ────────────────────────────────────────────────────────────────────
def step_predict_challenge(dataset_name, trained_results, softmax_w, device=None):
    """Challenge inference.  Uses the FINAL model (all-5-splits), which is
    correct here — we want maximum data for the challenge submission.
    softmax_w passed directly from step_evaluate_classifiers.
    """
    device = device or config.DEVICE

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np    = X_real_df.values.astype(np.float32)
    sample_ids   = list(X_real_df.index)

    proxy_ckpt = os.path.join(config.SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        print("  Loading existing target proxy...")
        model, diff_trainer = load_target_proxy(dataset_name, device=device)
        X_syn     = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        print("  Training target proxy on challenge synthetic data...")
        model, diff_trainer, scaler = train_target_proxy(dataset_name, device=device)

    y_int    = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)
    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

    raw_challenge = extract_loss_features(
        model, diff_trainer, X_scaled, y_int,
        t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device
    )

    ds         = config.DATASETS[dataset_name]
    all_scores = {}

    for clf_name, entry_tuple in trained_results.items():
        clf, _, t_indices, noise_budget = entry_tuple[:4]

        X_feat = summarize_features(
            slice_raw_features(raw_challenge, t_indices, noise_budget),
            noise_budget, len(t_indices)
        )
        scores = get_classifier(clf_name).predict(clf, X_feat)
        all_scores[clf_name] = scores

        out_path = os.path.join(
            ds["challenge_dir"],
            f"synthetic_data_1_predictions_synth_shadow_{clf_name}.csv",
        )
        pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
        print(f"  [{clf_name.upper()}] t_indices={[config.T_SUPERSET[i] for i in t_indices]}  "
              f"n_noise={noise_budget}  →  {out_path}")

    ensemble_scores = np.zeros(len(sample_ids), dtype=np.float64)
    print(f"\n  [ENSEMBLE] temperature={config.ENSEMBLE_TEMPERATURE}  "
          f"min_auc_gate={config.ENSEMBLE_MIN_AUC_GATE}")
    for clf_name, w in softmax_w.items():
        ensemble_scores += all_scores[clf_name] * w
        print(f"  [ENSEMBLE]   {clf_name.upper():<6}  weight={w:.4f}")

    ens_path = os.path.join(
        ds["challenge_dir"],
        "synthetic_data_1_predictions_synth_shadow_ensemble.csv",
    )
    pd.DataFrame({"sample_id": sample_ids, "score": ensemble_scores}).to_csv(ens_path, index=False)
    print(f"  [ENSEMBLE] → {ens_path}")


# ── Step 6 ────────────────────────────────────────────────────────────────────
def generate_report(dataset_name, trained_results, split_summary, ensemble_weights):
    out_dir   = config.MIA_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path  = os.path.join(out_dir, f"synth_shadow_report_{dataset_name}_{ts}.txt")
    json_path = os.path.join(out_dir, f"synth_shadow_report_{dataset_name}_{ts}.json")

    n_splits    = config.NUM_SPLITS
    clf_names   = list(trained_results.keys())
    fpr_targets = [0.01, 0.05, 0.10, 0.20]

    report_data = {
        "dataset":               dataset_name,
        "timestamp":             ts,
        "profile":               config.ACTIVE_PROFILE,
        "T_superset":            list(config.T_SUPERSET),
        "N_noise":               config.N_NOISE,
        "val_method":            "LOSO-CV (EVAL_ZONE S4-S5 only, leak-free; trains on all other splits)",
        "mlp_T_list":            _MLP_T_LIST,
        "mlp_N_noise":           _MLP_N_NOISE,
        "mlp_hidden_dim":        _MLP_HIDDEN_DIM,
        "mlp_epochs":            _MLP_EPOCHS,
        "ensemble_temperature":  config.ENSEMBLE_TEMPERATURE,
        "ensemble_min_auc_gate": config.ENSEMBLE_MIN_AUC_GATE,
        "classifiers":           {},
        "ensemble":              {},
    }

    lines = []
    lines.append("=" * 80)
    lines.append(f"  SYNTH-SHADOW MIA REPORT — {dataset_name}  [{ts}]")
    lines.append(f"  Profile: {config.ACTIVE_PROFILE}  |  T_SUPERSET={config.T_SUPERSET}")
    lines.append(
        f"  N_NOISE={config.N_NOISE}  |  "
        f"Val method: LOSO-CV (EVAL_ZONE only; trains on all non-val splits)  |  "
        f"Optuna={'ON (trees only)' if config.OPTUNA_ENABLED else 'OFF'}  |  "
        f"ensemble_temperature={config.ENSEMBLE_TEMPERATURE}  "
        f"min_auc_gate={config.ENSEMBLE_MIN_AUC_GATE}"
    )
    lines.append(
        f"  MLP (fixed): T={_MLP_T_LIST}  N_NOISE={_MLP_N_NOISE}  "
        f"hidden_dim={_MLP_HIDDEN_DIM}  epochs={_MLP_EPOCHS}"
    )
    lines.append("=" * 80)

    for clf_name in clf_names:
        entry_tuple = trained_results[clf_name]
        val_tpr     = entry_tuple[4]
        val_auc     = entry_tuple[5]
        selected_ts = entry_tuple[6]
        loso_folds  = entry_tuple[7]
        splits      = split_summary.get(clf_name, {})
        optuna_tag  = " [fixed]" if clf_name in _NO_OPTUNA_CLASSIFIERS else " [Optuna]"

        lines.append(f"\n  ── {clf_name.upper()}{optuna_tag}  (T-subset={selected_ts}) ──")
        lines.append(
            f"  {'':8}  {'@1%FPR':>8}  {'@5%FPR':>8}  {'@10%FPR':>9}  {'@20%FPR':>9}  {'AUC':>7}"
        )
        lines.append(
            f"  {'[LOSO]':<8}  {'':>8}  {'':>8}  {val_tpr:9.4f}  {'':>9}  {val_auc:7.4f}  "
            f"← LOSO mean (EVAL_ZONE, {len(loso_folds)} folds, per-fold={[f'{t:.3f}' for t in loso_folds]})"
        )

        split_tprs_10 = []
        split_aucs    = []
        multi_fpr_agg = {t: [] for t in fpr_targets}

        _rpt_keys = sorted(splits.keys())
        for s in _rpt_keys:
            sp   = splits.get(s, {})
            mfp  = sp.get("multi_fpr", {t: 0.0 for t in fpr_targets})
            auc  = sp.get("auc", 0.0)
            lines.append(
                f"  [S{s} EVAL]  "
                f"{mfp.get(0.01, 0.0):8.4f}  "
                f"{mfp.get(0.05, 0.0):8.4f}  "
                f"{mfp.get(0.10, 0.0):9.4f}  "
                f"{mfp.get(0.20, 0.0):9.4f}  "
                f"{auc:7.4f}"
            )
            split_tprs_10.append(mfp.get(0.10, 0.0))
            split_aucs.append(auc)
            for t in fpr_targets:
                multi_fpr_agg[t].append(mfp.get(t, 0.0))

        mean_mfp = {t: float(np.mean(multi_fpr_agg[t])) if len(multi_fpr_agg[t]) else 0.0
                    for t in fpr_targets}
        lines.append(
            f"  {'[MEAN]':<8}  "
            f"{mean_mfp[0.01]:8.4f}  "
            f"{mean_mfp[0.05]:8.4f}  "
            f"{mean_mfp[0.10]:9.4f}  "
            f"{mean_mfp[0.20]:9.4f}  "
            f"{float(np.mean(split_aucs)) if split_aucs else 0.0:7.4f}"
        )

        report_data["classifiers"][clf_name] = {
            "val_tpr_10fpr":        round(float(val_tpr), 6),
            "val_auc":              round(float(val_auc), 6),
            "loso_fold_tprs":       [round(float(t), 6) for t in loso_folds],
            "mean_split_tpr_10fpr": round(float(np.mean(split_tprs_10)) if split_tprs_10 else 0.0, 6),
            "mean_split_auc":       round(float(np.mean(split_aucs)) if split_aucs else 0.0, 6),
            "mean_multi_fpr":       {str(t): round(mean_mfp[t], 6) for t in fpr_targets},
            "selected_ts":          [int(t) for t in selected_ts],
            "optuna":               clf_name not in _NO_OPTUNA_CLASSIFIERS,
            "splits": {
                str(s): {
                    "multi_fpr": {
                        str(t): round(float(
                            splits.get(s, {}).get("multi_fpr", {}).get(t, 0.0)
                        ), 6)
                        for t in fpr_targets
                    },
                    "auc": round(float(splits.get(s, {}).get("auc", 0.0)), 6),
                }
                for s in _rpt_keys
            },
        }

    ens_splits   = split_summary.get("ensemble", {})
    ens_tprs_10  = []
    ens_aucs     = []
    ens_mfpr_agg = {t: [] for t in fpr_targets}

    lines.append(f"\n  ── ENSEMBLE  (Weighted Blend) ──")
    lines.append(
        f"  {'':8}  {'@1%FPR':>8}  {'@5%FPR':>8}  {'@10%FPR':>9}  {'@20%FPR':>9}  {'AUC':>7}"
    )

    _ens_rpt_keys = sorted(ens_splits.keys()) if ens_splits else []
    for s in _ens_rpt_keys:
        sp  = ens_splits.get(s, {})
        mfp = sp.get("multi_fpr", {t: 0.0 for t in fpr_targets})
        auc = sp.get("auc", 0.0)
        lines.append(
            f"  [S{s} EVAL]  "
            f"{mfp.get(0.01, 0.0):8.4f}  "
            f"{mfp.get(0.05, 0.0):8.4f}  "
            f"{mfp.get(0.10, 0.0):9.4f}  "
            f"{mfp.get(0.20, 0.0):9.4f}  "
            f"{auc:7.4f}"
        )
        ens_tprs_10.append(mfp.get(0.10, 0.0))
        ens_aucs.append(auc)
        for t in fpr_targets:
            ens_mfpr_agg[t].append(mfp.get(t, 0.0))

    ens_mean_mfp = {t: float(np.mean(ens_mfpr_agg[t])) if len(ens_mfpr_agg[t]) else 0.0
                    for t in fpr_targets}
    lines.append(
        f"  {'[MEAN]':<8}  "
        f"{ens_mean_mfp[0.01]:8.4f}  "
        f"{ens_mean_mfp[0.05]:8.4f}  "
        f"{ens_mean_mfp[0.10]:9.4f}  "
        f"{ens_mean_mfp[0.20]:9.4f}  "
        f"{float(np.mean(ens_aucs)) if ens_aucs else 0.0:7.4f}"
    )

    report_data["ensemble"]["performance"] = {
        "mean_split_tpr_10fpr": round(float(np.mean(ens_tprs_10)) if ens_tprs_10 else 0.0, 6),
        "mean_split_auc":       round(float(np.mean(ens_aucs)) if ens_aucs else 0.0, 6),
        "mean_multi_fpr":       {str(t): round(ens_mean_mfp[t], 6) for t in fpr_targets},
        "splits": {
            str(s): {
                "multi_fpr": {
                    str(t): round(float(
                        ens_splits.get(s, {}).get("multi_fpr", {}).get(t, 0.0)
                    ), 6)
                    for t in fpr_targets
                },
                "auc": round(float(ens_splits.get(s, {}).get("auc", 0.0)), 6),
            }
            for s in _ens_rpt_keys
        },
    }

    lines.append("\n" + "─" * 80)
    lines.append(
        f"  ENSEMBLE WEIGHTS  "
        f"(softmax T={config.ENSEMBLE_TEMPERATURE}, "
        f"AUC gate={config.ENSEMBLE_MIN_AUC_GATE})"
    )
    lines.append(f"  {'CLF':<8}  {'weight':>8}  {'status':>8}  {'optuna':>8}")
    for clf_name, w in ensemble_weights.items():
        gated   = w < 1e-6
        status  = "GATED" if gated else "active"
        opt_tag = "no" if clf_name in _NO_OPTUNA_CLASSIFIERS else "yes"
        lines.append(f"  {clf_name.upper():<8}  {w:8.4f}  {status:>8}  {opt_tag:>8}")
        report_data["ensemble"][clf_name] = {
            "weight": round(float(w), 6),
            "gated":  bool(gated),
        }

    lines.append("\n" + "=" * 80)

    txt_content = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(txt_content)
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n  [REPORT] → {txt_path}")
    print(f"  [REPORT] → {json_path}")
    print(txt_content)


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(dataset_name, device=None):
    device = device or config.DEVICE
    print("\n" + "=" * 70)
    print(f"SYNTH-SHADOW PIPELINE (SUPERSET): {dataset_name}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"STEP 1: Shadow models ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Superset feature extraction  "
          f"T={config.T_SUPERSET}  N_NOISE={config.N_NOISE}  ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Train classifiers  "
          f"(MLP: fixed | trees: Optuna {'ON' if config.OPTUNA_ENABLED else 'OFF'})  "
          f"({dataset_name})")
    print(f"        Val method: EVAL_ZONE LOSO-CV (Optuna sealed to S1-S3)")
    print("=" * 70)
    trained_results = step_train_classifiers(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation + ensemble  ({dataset_name})")
    print(f"        (All numbers are EVAL_ZONE LOSO held-out — no final-model re-scoring)")
    print("=" * 70)
    split_summary, ensemble_weights = step_evaluate_classifiers(
        dataset_name, trained_results
    )

    print("\n" + "=" * 70)
    print(f"STEP 5: Challenge predictions ({dataset_name})")
    print("=" * 70)
    step_predict_challenge(dataset_name, trained_results, ensemble_weights, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 6: Report ({dataset_name})")
    print("=" * 70)
    generate_report(dataset_name, trained_results, split_summary, ensemble_weights)

    return trained_results


# ── CLI ───────────────────────────────────────────────────────────────────────
# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="COMBINED")
    parser.add_argument(
        "--force",
        type=str,
        default="",
        help="Comma-separated list of stages to force-rerun: "
             "shadows,features,classifier,challenge,all",
    )
    args = parser.parse_args()

    # Push --force into config so _force() picks it up everywhere
    config.FORCE_STAGES = [s.strip() for s in args.force.split(",") if s.strip()]

    run_full_pipeline(args.dataset)
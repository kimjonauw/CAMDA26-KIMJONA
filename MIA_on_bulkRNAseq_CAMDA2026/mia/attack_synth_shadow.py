"""Synth-shadow MIA pipeline with superset slicing + Optuna joint search.

KEY FIXES:
  1. MLP Checkpointing: Sealed using an internal GroupShuffleSplit ES split.
  2. Optuna StratifiedGroupKFold: Prevents hyperparameter tuning from memorizing 
     subject identities across pooled splits.
  3. Subject-Disjoint LOSO: Training strips eval-zone subjects completely.
  4. Reference Calibration: Normalizes raw loss trajectories against a pure 
     non-member baseline to isolate memorization from biological variance.
"""

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from . import config
from .data_utils import (
    load_real_data, load_nd_synthetic, load_challenge_synthetic,
    get_nd_membership_labels, fit_quantile_scaler, load_reference_data
)
from .shadow_model import (
    train_shadow_model, build_model, build_diffusion_trainer,
    train_target_proxy, load_target_proxy, load_shadow_scaler, _scaler_path,        
)
from .loss_features import (
    extract_loss_features, slice_raw_features, summarize_features, prepare_features,
)
from .classifier import (
    get_classifier, _tpr_at_fpr, tpr_at_fpr_multi,
    compute_softmax_weights, train_classifier, load_classifier,
)

_NO_OPTUNA_CLASSIFIERS = {"mlp"}
_MLP_T_LIST     = [1, 2, 5, 10, 20, 50, 100, 200]
_MLP_N_NOISE    = 100
_MLP_HIDDEN_DIM = 64
_MLP_EPOCHS     = 2000

def _force(stage): return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES
def _shadow_model_dir(dn):  return os.path.join(config.SYNTH_SHADOW_MODEL_DIR, dn)
def _features_dir(dn):      return os.path.join(config.SYNTH_SHADOW_FEATURES_DIR, dn)
def _classifier_dir(cn):    return os.path.join(config.SYNTH_SHADOW_CLASSIFIER_DIR, cn)
def _optuna_dir(dn, cn):    return os.path.join(config.SYNTH_SHADOW_OPTUNA_DIR, dn, cn)

T_GROUPS = {
    "very_early": [0, 1, 2],
    "early":      [3, 4, 5, 6],
    "mid":        [7, 8, 9],
    "late":       [10, 11, 12],
    "very_late":  [13, 14],
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

_SUGGEST = {"xgb": _suggest_xgb, "rf": _suggest_rf, "lgbm": _suggest_lgbm, "cat": _suggest_cat}


def _make_objective(clf_name, raw_opt, y_opt, groups_opt, n_folds):
    entry   = get_classifier(clf_name)
    suggest = _SUGGEST[clf_name]
    gkf     = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    def objective(trial):
        t_indices    = _build_t_indices(trial)
        noise_budget = trial.suggest_int("n_noise", 150, config.N_NOISE, step=50)
        n_t          = len(t_indices)

        X_sliced = slice_raw_features(raw_opt, t_indices, noise_budget)
        X_feat   = summarize_features(X_sliced, noise_budget, n_t)
        hparams  = suggest(trial)

        fold_tprs = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(gkf.split(X_feat, y_opt, groups=groups_opt)):
            X_tr, y_tr = X_feat[tr_idx], y_opt[tr_idx]
            id_tr      = groups_opt[tr_idx]
            X_vl, y_vl = X_feat[vl_idx], y_opt[vl_idx]
            tmp_dir = f"/tmp/optuna_{clf_name}_t{trial.number}_f{fold_idx}"
            try:
                clf, _  = entry.train(X_tr, y_tr, id_tr, X_vl, y_vl, tmp_dir, hparams=hparams)
                scores  = entry.predict(clf, X_vl)
                fold_tprs.append(_tpr_at_fpr(y_vl.astype(int), scores))
            except Exception as e:
                print(f"  [Optuna/{clf_name}] trial={trial.number} error: {e}")
                fold_tprs.append(0.0)

        return float(np.mean(fold_tprs))
    return objective


def run_optuna(clf_name, raw_opt, y_opt, groups_opt, dataset_name):
    out_dir    = _optuna_dir(dataset_name, clf_name)
    os.makedirs(out_dir, exist_ok=True)
    best_path  = os.path.join(out_dir, "best_params.json")
    db_path    = os.path.join(out_dir, "study.db")
    storage    = config.OPTUNA_STORAGE or f"sqlite:///{db_path}"
    study_name = f"{dataset_name}_{clf_name}_{config.ACTIVE_PROFILE}"

    if _force("classifier"):
        if os.path.exists(best_path): os.remove(best_path)
        try: optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError: pass
        if not config.OPTUNA_STORAGE and os.path.exists(db_path): os.remove(db_path)

    if not _force("classifier") and os.path.exists(best_path):
        with open(best_path) as f: best = json.load(f)
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
    objective = _make_objective(clf_name, raw_opt, y_opt, groups_opt, n_folds)
    study.optimize(objective, n_trials=n_trials, timeout=config.OPTUNA_TIMEOUT, n_jobs=1, show_progress_bar=False)

    best = study.best_params
    print(f"  [Optuna/{clf_name.upper()}] Best mean TPR@10%FPR = {study.best_value:.4f}")
    print(f"  [Optuna/{clf_name.upper()}] Best params: {best}")
    with open(best_path, "w") as f: json.dump(best, f, indent=2)
    return best


def _params_to_t_indices_and_noise(best_params):
    t_indices = []
    for name in T_GROUP_NAMES:
        if best_params.get(f"use_{name}", False): t_indices.extend(T_GROUPS[name])
    if len(t_indices) < 3: t_indices = sorted(set(t_indices) | set(T_GROUPS["early"] + T_GROUPS["mid"]))
    else: t_indices = sorted(set(t_indices))
    noise_budget = int(best_params.get("n_noise", config.N_NOISE))
    return t_indices, noise_budget


def _extract_clf_hparams(best_params, clf_name):
    return {k: v for k, v in best_params.items() if not k.startswith("use_") and k != "n_noise"}


def calibrate_raw_features(raw, raw_ref, n_t, n_noise):
    """Z-score raw features per timestep using the reference non-member distribution.
    This effectively subtracts intrinsic biological difficulty and shadow batch effects.
    """
    if raw_ref is None or len(raw_ref) == 0:
        return raw
    r   = raw.reshape(raw.shape[0], n_t, n_noise)
    ref = raw_ref.reshape(raw_ref.shape[0], n_t, n_noise)
    
    # Compute population mean/std across reference samples AND noise vectors at each t
    ref_mean = ref.mean(axis=(0, 2)).reshape(1, n_t, 1)
    ref_std  = ref.std(axis=(0, 2)).reshape(1, n_t, 1)
    
    calibrated = (r - ref_mean) / (ref_std + 1e-8)
    return calibrated.reshape(raw.shape[0], n_t * n_noise)


def step_train_shadows(dataset_name, splits=None, device=None):
    splits    = splits or list(range(1, config.NUM_SPLITS + 1))
    device    = device or config.DEVICE
    save_dir  = _shadow_model_dir(dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    force = _force("shadows")
    for s in splits:
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path): continue
        X_syn, y_str = load_nd_synthetic(dataset_name, s)
        train_shadow_model(X_train=X_syn, save_path=save_path, split_no=s, device=device, y_str=y_str, dataset_name=dataset_name)


def step_extract_features(dataset_name, splits=None, device=None):
    splits    = splits or list(range(1, config.NUM_SPLITS + 1))
    device    = device or config.DEVICE
    feat_dir  = _features_dir(dataset_name)
    os.makedirs(feat_dir, exist_ok=True)
    model_dir = _shadow_model_dir(dataset_name)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np    = X_real_df.values.astype(np.float32)
    sample_ids   = list(X_real_df.index)
    y_int        = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    # Load reference data (if dataset supports it, e.g. COMBINED)
    X_ref_df = load_reference_data(dataset_name)
    X_ref_np = X_ref_df.values.astype(np.float32) if X_ref_df is not None else None
    y_ref_int = np.full(len(X_ref_np), config.DUMMY_LABEL, dtype=np.int64) if X_ref_np is not None else None

    force = _force("features")
    for s in splits:
        out_path = os.path.join(feat_dir, f"features_split_{s}.npz")
        
        # If forcing features, we recalculate. Otherwise check if it exists and has the reference data.
        if not force and os.path.exists(out_path):
            try:
                d = np.load(out_path)
                if d["features"].shape[1] == len(config.T_SUPERSET) * config.N_NOISE:
                    if X_ref_np is None or "ref_features" in d:
                        continue  # Safe to skip
            except: pass

        model = build_model(config.UNCONDITIONAL_NUM_CLASSES, device)
        ckpt  = os.path.join(model_dir, f"shadow_split_{s}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        diff_trainer = build_diffusion_trainer(device)
        scaler = load_shadow_scaler(ckpt)

        X_scaled  = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
        features  = extract_loss_features(model, diff_trainer, X_scaled, y_int, t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device)
        
        ref_features = None
        if X_ref_np is not None:
            X_ref_scaled = scaler.transform(X_ref_np.astype(np.float64)).astype(np.float32)
            ref_features = extract_loss_features(model, diff_trainer, X_ref_scaled, y_ref_int, t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device)
            
        _, y_member = get_nd_membership_labels(dataset_name, s)

        if ref_features is not None:
            np.savez(out_path, features=features, y_member=y_member, y_label_int=y_int, sample_ids=sample_ids, ref_features=ref_features)
        else:
            np.savez(out_path, features=features, y_member=y_member, y_label_int=y_int, sample_ids=sample_ids)


def _subject_disjoint_loso(clf_name, entry, raw_splits, y_splits, id_splits,
                           t_indices, noise_budget, clf_hparams, eval_indices):
    n_splits    = len(raw_splits)
    fold_tprs   = []
    fold_aucs   = []
    fold_auprs  = []
    fold_scores = {}
    fold_y      = {}

    for val_s in eval_indices:
        prev_s = (val_s - 1) % n_splits
        
        nonmember_val_subjects = set(id_splits[val_s][y_splits[val_s] == 0])
        member_val_subjects    = set(id_splits[prev_s][y_splits[prev_s] == 0])
        val_subjects           = nonmember_val_subjects | member_val_subjects

        train_idx   = [i for i in range(n_splits) if i not in (val_s, prev_s)]
        raw_tr_pool = np.concatenate([raw_splits[i] for i in train_idx])
        y_tr_pool   = np.concatenate([y_splits[i]   for i in train_idx])
        id_tr_pool  = np.concatenate([id_splits[i]  for i in train_idx])

        train_mask = np.array([sid not in val_subjects for sid in id_tr_pool])
        raw_tr = raw_tr_pool[train_mask]
        y_tr   = y_tr_pool[train_mask]
        id_tr  = id_tr_pool[train_mask]

        val_mask = np.array([sid in val_subjects for sid in id_splits[val_s]])
        raw_vl = raw_splits[val_s][val_mask]
        y_vl   = y_splits[val_s][val_mask]

        X_tr = summarize_features(slice_raw_features(raw_tr, t_indices, noise_budget), noise_budget, len(t_indices))
        X_vl = summarize_features(slice_raw_features(raw_vl, t_indices, noise_budget), noise_budget, len(t_indices))

        try:
            clf, _ = entry.train(X_tr, y_tr, id_tr, X_vl, y_vl, f"/tmp/loso_{clf_name}_fold{val_s}", hparams=clf_hparams)
            scores = entry.predict(clf, X_vl)

            tpr  = _tpr_at_fpr(y_vl.astype(int), scores)
            auc  = roc_auc_score(y_vl.astype(int), scores)
            aupr = average_precision_score(y_vl.astype(int), scores)

            print(f"    [Disjoint LOSO S{val_s+1}]  Train rows={len(y_tr)}  Eval rows={len(y_vl)}  TPR@10%={tpr:.4f}  AUC={auc:.4f}  AUPR={aupr:.4f}")

            fold_tprs.append(tpr)
            fold_aucs.append(auc)
            fold_auprs.append(aupr)
            fold_scores[val_s] = scores
            fold_y[val_s]      = y_vl

        except Exception as e:
            print(f"    [LOSO S{val_s+1}] error: {e}")
            fold_tprs.append(0.0)
            fold_aucs.append(0.5)
            fold_auprs.append(0.0)
            fold_scores[val_s] = np.zeros(len(y_vl))
            fold_y[val_s]      = y_vl

    mean_tpr  = float(np.mean(fold_tprs))
    mean_auc  = float(np.mean(fold_aucs))
    mean_aupr = float(np.mean(fold_auprs))
    return mean_tpr, mean_auc, mean_aupr, fold_tprs, fold_scores, fold_y, fold_aucs, fold_auprs


def step_train_classifiers(dataset_name, device=None):
    device   = device or config.DEVICE
    feat_dir = _features_dir(dataset_name)
    n_splits = config.NUM_SPLITS

    raw_splits, y_splits, id_splits = [], [], []
    for s in range(1, n_splits + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        raw_s = d["features"]
        
        # Instantly calibrate raw features using the shadow-specific reference baseline
        if "ref_features" in d and d["ref_features"] is not None and len(d["ref_features"].shape) > 0:
            raw_s = calibrate_raw_features(raw_s, d["ref_features"], len(config.T_SUPERSET), config.N_NOISE)

        raw_splits.append(raw_s)
        y_splits.append(d["y_member"])
        id_splits.append(d["sample_ids"])

    raw_all = np.concatenate(raw_splits)
    y_all   = np.concatenate(y_splits)
    id_all  = np.concatenate(id_splits)

    import math
    n_opt_splits  = math.ceil(n_splits * 0.6)
    opt_indices   = list(range(n_opt_splits))
    eval_indices  = list(range(n_opt_splits, n_splits))

    all_eval_subjects = set()
    for val_s in eval_indices:
        prev_s = (val_s - 1) % n_splits
        all_eval_subjects.update(id_splits[val_s][y_splits[val_s] == 0])
        all_eval_subjects.update(id_splits[prev_s][y_splits[prev_s] == 0])

    raw_opt_unfiltered = np.concatenate([raw_splits[i] for i in opt_indices])
    y_opt_unfiltered   = np.concatenate([y_splits[i]   for i in opt_indices])
    id_opt_unfiltered  = np.concatenate([id_splits[i]  for i in opt_indices])

    opt_mask = np.array([sid not in all_eval_subjects for sid in id_opt_unfiltered])
    raw_opt = raw_opt_unfiltered[opt_mask]
    y_opt   = y_opt_unfiltered[opt_mask]
    id_opt  = id_opt_unfiltered[opt_mask]

    results = {}
    for clf_name in config.ACTIVE_CLASSIFIERS:
        entry    = get_classifier(clf_name)
        save_dir = _classifier_dir(clf_name)

        if clf_name in _NO_OPTUNA_CLASSIFIERS:
            t_indices = [config.T_SUPERSET.index(t) for t in _MLP_T_LIST if t in config.T_SUPERSET]
            noise_budget = _MLP_N_NOISE
            clf_hparams  = {"hidden_dim": _MLP_HIDDEN_DIM, "epochs": _MLP_EPOCHS, "dropout": config.MLP_DROPOUT, "weight_decay": config.MLP_WEIGHT_DECAY, "lr": config.MLP_LR, "batch_size": config.MLP_BATCH_SIZE}
            selected_ts = [config.T_SUPERSET[i] for i in t_indices]
        elif config.OPTUNA_ENABLED:
            best_params             = run_optuna(clf_name, raw_opt, y_opt, id_opt, dataset_name)
            t_indices, noise_budget = _params_to_t_indices_and_noise(best_params)
            clf_hparams             = _extract_clf_hparams(best_params, clf_name)
            selected_ts             = [config.T_SUPERSET[i] for i in t_indices]
        else:
            t_indices, noise_budget, clf_hparams, selected_ts = list(range(len(config.T_SUPERSET))), config.N_NOISE, None, list(config.T_SUPERSET)

        (val_tpr, val_auc, val_aupr, loso_fold_tprs, loso_fold_scores, loso_fold_y, loso_fold_aucs, loso_fold_auprs) = _subject_disjoint_loso(
            clf_name, entry, raw_splits, y_splits, id_splits, t_indices, noise_budget, clf_hparams, eval_indices=eval_indices
        )

        X_all = summarize_features(slice_raw_features(raw_all, t_indices, noise_budget), noise_budget, len(t_indices))
        clf, _ = entry.train(X_all, y_all, id_all, X_all[:2], y_all[:2], save_dir, hparams=clf_hparams)

        meta = {
            "t_indices": t_indices, 
            "noise_budget": noise_budget, 
            "clf_hparams": clf_hparams or {}, 
            "selected_ts": selected_ts, 
            "loso_fold_tprs": loso_fold_tprs, 
            "loso_fold_aucs": loso_fold_aucs, 
            "loso_fold_auprs": loso_fold_auprs, 
            "loso_mean_tpr": val_tpr, 
            "loso_mean_auc": val_auc,
            "loso_mean_aupr": val_aupr
        }
        with open(os.path.join(save_dir, "signal_params.json"), "w") as f: json.dump(meta, f, indent=2)

        results[clf_name] = (clf, None, t_indices, noise_budget, val_tpr, val_auc, selected_ts, loso_fold_tprs, loso_fold_scores, loso_fold_y, loso_fold_aucs, val_aupr, loso_fold_auprs)
    return results


def step_evaluate_classifiers(dataset_name, trained_results):
    n_splits    = config.NUM_SPLITS
    fpr_targets = [0.01, 0.05, 0.10, 0.20]
    summary     = {clf_name: {} for clf_name in trained_results}
    summary["ensemble"] = {}

    ref_clf          = next(iter(trained_results))
    loso_keys_0based = sorted(trained_results[ref_clf][9].keys())

    split_scores = {}
    split_y      = {}

    for val_s in loso_keys_0based:
        global_s = val_s + 1
        split_scores[global_s] = {}
        split_y[global_s] = trained_results[ref_clf][9][val_s]

    for clf_name, entry_tuple in trained_results.items():
        (clf, _, t_indices, noise_budget, val_tpr, val_auc, selected_ts, loso_fold_tprs, loso_fold_scores, loso_fold_y, loso_fold_aucs, val_aupr, loso_fold_auprs) = entry_tuple

        for idx, val_s in enumerate(loso_keys_0based):
            global_s = val_s + 1
            scores   = loso_fold_scores[val_s]
            y_member = loso_fold_y[val_s]
            split_scores[global_s][clf_name] = scores

            tpr       = _tpr_at_fpr(y_member.astype(int), scores)
            auc       = loso_fold_aucs[idx]
            aupr      = loso_fold_auprs[idx]
            multi_fpr = tpr_at_fpr_multi(y_member.astype(int), scores)
            summary[clf_name][global_s] = {"tpr": tpr, "auc": auc, "aupr": aupr, "multi_fpr": multi_fpr}

    loso_aucs_overall = {n: float(trained_results[n][5]) for n in trained_results}
    final_softmax_w = compute_softmax_weights(loso_aucs_overall, temperature=config.ENSEMBLE_TEMPERATURE, min_auc_gate=config.ENSEMBLE_MIN_AUC_GATE)

    for idx, val_s in enumerate(loso_keys_0based):
        global_s = val_s + 1
        y_member = split_y[global_s]
        
        fold_aucs = {}
        for clf_name in trained_results:
            aucs_excluding_current = [
                trained_results[clf_name][10][i] 
                for i, s in enumerate(loso_keys_0based) if s != val_s
            ]
            fold_aucs[clf_name] = float(np.mean(aucs_excluding_current)) if aucs_excluding_current else 0.5
            
        fold_softmax_w = compute_softmax_weights(fold_aucs, temperature=config.ENSEMBLE_TEMPERATURE, min_auc_gate=config.ENSEMBLE_MIN_AUC_GATE)

        ens_scores = np.zeros(len(y_member), dtype=np.float64)
        for clf_name, w in fold_softmax_w.items(): 
            ens_scores += split_scores[global_s][clf_name] * w
            
        tpr       = _tpr_at_fpr(y_member.astype(int), ens_scores)
        auc       = roc_auc_score(y_member.astype(int), ens_scores) if len(np.unique(y_member)) > 1 else 0.5
        aupr      = average_precision_score(y_member.astype(int), ens_scores) if len(np.unique(y_member)) > 1 else 0.0
        multi_fpr = tpr_at_fpr_multi(y_member.astype(int), ens_scores)
        summary["ensemble"][global_s] = {"tpr": tpr, "auc": auc, "aupr": aupr, "multi_fpr": multi_fpr}

    return summary, final_softmax_w


def step_predict_challenge(dataset_name, trained_results, softmax_w, device=None):
    device = device or config.DEVICE
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np    = X_real_df.values.astype(np.float32)
    sample_ids   = list(X_real_df.index)
    
    # Load reference data
    X_ref_df = load_reference_data(dataset_name)
    X_ref_np = X_ref_df.values.astype(np.float32) if X_ref_df is not None else None

    proxy_ckpt = os.path.join(config.SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        model, diff_trainer = load_target_proxy(dataset_name, device=device)
        X_syn     = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        model, diff_trainer, scaler = train_target_proxy(dataset_name, device=device)

    y_int    = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)
    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
    raw_challenge = extract_loss_features(model, diff_trainer, X_scaled, y_int, t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device)

    # Apply reference calibration to challenge extractions
    if X_ref_np is not None:
        y_ref_int = np.full(len(X_ref_np), config.DUMMY_LABEL, dtype=np.int64)
        X_ref_scaled = scaler.transform(X_ref_np.astype(np.float64)).astype(np.float32)
        ref_challenge = extract_loss_features(model, diff_trainer, X_ref_scaled, y_ref_int, t_list=config.T_SUPERSET, n_noise=config.N_NOISE, device=device)
        raw_challenge = calibrate_raw_features(raw_challenge, ref_challenge, len(config.T_SUPERSET), config.N_NOISE)

    ds         = config.DATASETS[dataset_name]
    all_scores = {}

    for clf_name, entry_tuple in trained_results.items():
        clf, _, t_indices, noise_budget = entry_tuple[:4]
        X_feat = summarize_features(slice_raw_features(raw_challenge, t_indices, noise_budget), noise_budget, len(t_indices))
        scores = get_classifier(clf_name).predict(clf, X_feat)
        all_scores[clf_name] = scores
        out_path = os.path.join(ds["challenge_dir"], f"synthetic_data_1_predictions_synth_shadow_{clf_name}.csv")
        pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)

    ensemble_scores = np.zeros(len(sample_ids), dtype=np.float64)
    for clf_name, w in softmax_w.items(): ensemble_scores += all_scores[clf_name] * w

    ens_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions_synth_shadow_ensemble.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": ensemble_scores}).to_csv(ens_path, index=False)


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
        "val_method":            "Subject-Disjoint LOSO (EVAL_ZONE S4-S5 only, leak-free)",
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
        f"Val method: Subject-Disjoint LOSO (EVAL_ZONE only)  |  "
        f"Optuna={'ON (trees only)' if config.OPTUNA_ENABLED else 'OFF'}  |  "
        f"ensemble_temperature={config.ENSEMBLE_TEMPERATURE}  "
        f"min_auc_gate={config.ENSEMBLE_MIN_AUC_GATE}"
    )
    lines.append("=" * 80)

    for clf_name in clf_names:
        entry_tuple = trained_results[clf_name]
        val_tpr     = entry_tuple[4]
        val_auc     = entry_tuple[5]
        selected_ts = entry_tuple[6]
        loso_folds  = entry_tuple[7]
        val_aupr    = entry_tuple[11]
        splits      = split_summary.get(clf_name, {})
        optuna_tag  = " [fixed]" if clf_name in _NO_OPTUNA_CLASSIFIERS else " [Optuna]"

        lines.append(f"\n  ── {clf_name.upper()}{optuna_tag}  (T-subset={selected_ts}) ──")
        lines.append(f"  {'':8}  {'@1%FPR':>8}  {'@5%FPR':>8}  {'@10%FPR':>9}  {'@20%FPR':>9}  {'AUC':>7}  {'AUPR':>7}")
        lines.append(
            f"  {'[LOSO]':<8}  {'':>8}  {'':>8}  {val_tpr:9.4f}  {'':>9}  {val_auc:7.4f}  {val_aupr:7.4f}  "
            f"← LOSO mean (EVAL_ZONE, {len(loso_folds)} folds, per-fold={[f'{t:.3f}' for t in loso_folds]})"
        )

        split_tprs_10 = []
        split_aucs    = []
        split_auprs   = []
        multi_fpr_agg = {t: [] for t in fpr_targets}

        _rpt_keys = sorted(splits.keys())
        for s in _rpt_keys:
            sp   = splits.get(s, {})
            mfp  = sp.get("multi_fpr", {t: 0.0 for t in fpr_targets})
            auc  = sp.get("auc", 0.0)
            aupr = sp.get("aupr", 0.0)
            lines.append(f"  [S{s} EVAL]  {mfp.get(0.01, 0.0):8.4f}  {mfp.get(0.05, 0.0):8.4f}  {mfp.get(0.10, 0.0):9.4f}  {mfp.get(0.20, 0.0):9.4f}  {auc:7.4f}  {aupr:7.4f}")
            split_tprs_10.append(mfp.get(0.10, 0.0))
            split_aucs.append(auc)
            split_auprs.append(aupr)
            for t in fpr_targets: multi_fpr_agg[t].append(mfp.get(t, 0.0))

        mean_mfp = {t: float(np.mean(multi_fpr_agg[t])) if len(multi_fpr_agg[t]) else 0.0 for t in fpr_targets}
        lines.append(f"  {'[MEAN]':<8}  {mean_mfp[0.01]:8.4f}  {mean_mfp[0.05]:8.4f}  {mean_mfp[0.10]:9.4f}  {mean_mfp[0.20]:9.4f}  {float(np.mean(split_aucs)) if split_aucs else 0.0:7.4f}  {float(np.mean(split_auprs)) if split_auprs else 0.0:7.4f}")

        report_data["classifiers"][clf_name] = {
            "val_tpr_10fpr":        round(float(val_tpr), 6),
            "val_auc":              round(float(val_auc), 6),
            "val_aupr":             round(float(val_aupr), 6),
            "loso_fold_tprs":       [round(float(t), 6) for t in loso_folds],
            "mean_split_tpr_10fpr": round(float(np.mean(split_tprs_10)) if split_tprs_10 else 0.0, 6),
            "mean_split_auc":       round(float(np.mean(split_aucs)) if split_aucs else 0.0, 6),
            "mean_split_aupr":      round(float(np.mean(split_auprs)) if split_auprs else 0.0, 6),
            "mean_multi_fpr":       {str(t): round(mean_mfp[t], 6) for t in fpr_targets},
            "selected_ts":          [int(t) for t in selected_ts],
            "splits": {
                str(s): {"multi_fpr": {str(t): round(float(splits.get(s, {}).get("multi_fpr", {}).get(t, 0.0)), 6) for t in fpr_targets},
                "auc": round(float(splits.get(s, {}).get("auc", 0.0)), 6),
                "aupr": round(float(splits.get(s, {}).get("aupr", 0.0)), 6)}
                for s in _rpt_keys
            },
        }

    ens_splits   = split_summary.get("ensemble", {})
    ens_tprs_10  = []
    ens_aucs     = []
    ens_auprs    = []
    ens_mfpr_agg = {t: [] for t in fpr_targets}

    lines.append(f"\n  ── ENSEMBLE  (Weighted Blend) ──")
    lines.append(f"  {'':8}  {'@1%FPR':>8}  {'@5%FPR':>8}  {'@10%FPR':>9}  {'@20%FPR':>9}  {'AUC':>7}  {'AUPR':>7}")

    _ens_rpt_keys = sorted(ens_splits.keys()) if ens_splits else []
    for s in _ens_rpt_keys:
        sp   = ens_splits.get(s, {})
        mfp  = sp.get("multi_fpr", {t: 0.0 for t in fpr_targets})
        auc  = sp.get("auc", 0.0)
        aupr = sp.get("aupr", 0.0)
        lines.append(f"  [S{s} EVAL]  {mfp.get(0.01, 0.0):8.4f}  {mfp.get(0.05, 0.0):8.4f}  {mfp.get(0.10, 0.0):9.4f}  {mfp.get(0.20, 0.0):9.4f}  {auc:7.4f}  {aupr:7.4f}")
        ens_tprs_10.append(mfp.get(0.10, 0.0))
        ens_aucs.append(auc)
        ens_auprs.append(aupr)
        for t in fpr_targets: ens_mfpr_agg[t].append(mfp.get(t, 0.0))

    ens_mean_mfp = {t: float(np.mean(ens_mfpr_agg[t])) if len(ens_mfpr_agg[t]) else 0.0 for t in fpr_targets}
    lines.append(f"  {'[MEAN]':<8}  {ens_mean_mfp[0.01]:8.4f}  {ens_mean_mfp[0.05]:8.4f}  {ens_mean_mfp[0.10]:9.4f}  {ens_mean_mfp[0.20]:9.4f}  {float(np.mean(ens_aucs)) if ens_aucs else 0.0:7.4f}  {float(np.mean(ens_auprs)) if ens_auprs else 0.0:7.4f}")

    report_data["ensemble"]["performance"] = {
        "mean_split_auc":       round(float(np.mean(ens_aucs)) if ens_aucs else 0.0, 6),
        "mean_split_aupr":      round(float(np.mean(ens_auprs)) if ens_auprs else 0.0, 6),
        "mean_multi_fpr":       {str(t): round(ens_mean_mfp[t], 6) for t in fpr_targets},
    }

    lines.append("\n" + "─" * 80)
    lines.append(f"  ENSEMBLE WEIGHTS  (softmax T={config.ENSEMBLE_TEMPERATURE}, AUC gate={config.ENSEMBLE_MIN_AUC_GATE})")
    lines.append(f"  {'CLF':<8}  {'weight':>8}  {'status':>8}  {'optuna':>8}")
    for clf_name, w in ensemble_weights.items():
        gated   = w < 1e-6
        status  = "GATED" if gated else "active"
        opt_tag = "no" if clf_name in _NO_OPTUNA_CLASSIFIERS else "yes"
        lines.append(f"  {clf_name.upper():<8}  {w:8.4f}  {status:>8}  {opt_tag:>8}")
        report_data["ensemble"][clf_name] = {"weight": round(float(w), 6), "gated": bool(gated)}
    lines.append("\n" + "=" * 80)

    txt_content = "\n".join(lines)
    with open(txt_path, "w") as f: f.write(txt_content)
    with open(json_path, "w") as f: json.dump(report_data, f, indent=2)

    print(f"\n  [REPORT] → {txt_path}")
    print(f"  [REPORT] → {json_path}")
    print(txt_content)


def run_full_pipeline(dataset_name, device=None):
    device = device or config.DEVICE
    step_train_shadows(dataset_name, device=device)
    step_extract_features(dataset_name, device=device)
    trained_results = step_train_classifiers(dataset_name, device=device)
    split_summary, ensemble_weights = step_evaluate_classifiers(dataset_name, trained_results)
    step_predict_challenge(dataset_name, trained_results, ensemble_weights, device=device)
    generate_report(dataset_name, trained_results, split_summary, ensemble_weights)
    return trained_results

if __name__ == "__main__":
    import sys, argparse
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="COMBINED")
    parser.add_argument("--force", type=str, default="", help="shadows,features,classifier,challenge,all")
    args = parser.parse_args()
    config.FORCE_STAGES = [s.strip() for s in args.force.split(",") if s.strip()]
    run_full_pipeline(args.dataset)
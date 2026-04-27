"""Membership inference classifiers: MLP, XGBoost, RandomForest, LightGBM, CatBoost."""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as SKLearnRF
from sklearn.metrics import roc_curve
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from . import config

torch.set_float32_matmul_precision('high')


def _tpr_at_fpr(y_true, y_score, fpr_target=0.10):
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    return float(tpr[valid][-1]) if valid.any() else 0.0


def tpr_at_fpr_multi(y_true, y_score, fpr_targets=(0.01, 0.05, 0.10, 0.20)):
    """Return dict of {fpr_target: tpr} for multiple FPR thresholds at once."""
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {t: 0.0 for t in fpr_targets}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    result = {}
    for target in fpr_targets:
        valid = fpr <= target
        result[target] = float(tpr[valid][-1]) if valid.any() else 0.0
    return result


def is_collapsed_classifier(y_true, y_score, acc_threshold=0.78, auc_threshold=0.72):
    """
    Detect dummy/collapsed classifiers that exploit class imbalance.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    acc = accuracy_score(y_true, (y_score > 0.5).astype(int))
    if len(np.unique(y_true)) < 2:
        return True
    auc = roc_auc_score(y_true, y_score)
    majority_prior = max(y_true.mean(), 1 - y_true.mean())
    acc_near_prior = abs(acc - majority_prior) < 0.005
    return bool(acc_near_prior and auc < auc_threshold)


def compute_softmax_weights(aucs: dict, temperature: float = 0.05, min_auc_gate: float = 0.72) -> dict:
    names  = list(aucs.keys())
    values = np.array([aucs[n] for n in names], dtype=float)
    gate   = (values >= min_auc_gate).astype(float)

    if gate.sum() == 0:
        warnings.warn(
            f"[ENSEMBLE] All classifiers below AUC gate={min_auc_gate:.2f}. "
            "Falling back to equal weights across all classifiers."
        )
        gate = np.ones(len(names), dtype=float)

    scaled   = values / temperature
    exp_v    = np.exp(scaled - scaled[gate > 0].max())
    exp_v   *= gate
    weights  = exp_v / exp_v.sum()

    return {n: float(w) for n, w in zip(names, weights)}


def _group_aware_es_split(X, y, groups, test_size=0.15, seed=None):
    """Forces internal Early Stopping to evaluate generalization on unseen subjects."""
    seed = seed or config.SEED
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, es_idx = next(gss.split(X, y, groups=groups))
    return X[fit_idx], X[es_idx], y[fit_idx], y[es_idx]


# ── MLP ───────────────────────────────────────────────────────────────────────

class MembershipMLP(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, dropout=0.0):
        super().__init__()
        input_dim  = input_dim  or config.MLP_INPUT_DIM
        hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_mlp(X_train, y_train, groups_train, X_report, y_report, save_dir, hparams=None):
    hp           = hparams or {}
    epochs       = hp.get("epochs",       config.MLP_EPOCHS)
    lr           = hp.get("lr",           config.MLP_LR)
    hidden_dim   = hp.get("hidden_dim",   config.MLP_HIDDEN_DIM)
    dropout      = hp.get("dropout",      config.MLP_DROPOUT)
    weight_decay = hp.get("weight_decay", config.MLP_WEIGHT_DECAY)
    batch_size   = hp.get("batch_size",   config.MLP_BATCH_SIZE)
    device       = config.DEVICE

    X_fit, X_es, y_fit, y_es = _group_aware_es_split(X_train, y_train, groups_train)

    n_pos   = int(y_fit.sum())
    n_neg   = len(y_fit) - n_pos
    pos_w   = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model     = MembershipMLP(input_dim=X_fit.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    train_ds     = TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_es_t       = torch.tensor(X_es, dtype=torch.float32).to(device)
    y_es_t       = torch.tensor(y_es, dtype=torch.float32).to(device)

    best_metric, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb).squeeze(1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            es_logits = model(X_es_t).squeeze(1)
            es_loss   = criterion(es_logits, y_es_t).item()
            es_scores = torch.sigmoid(es_logits).cpu().numpy()

        tpr = _tpr_at_fpr(y_es.astype(int), es_scores)

        if tpr > best_metric:
            best_metric = tpr
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) == epochs:
            print(f"  [MLP] epoch {epoch+1:4d}/{epochs}  train_loss={epoch_loss:.4f}  es_loss={es_loss:.4f}  ES_TPR@10%={tpr:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        report_logits = model(torch.tensor(X_report, dtype=torch.float32).to(device)).squeeze(1)
        report_scores = torch.sigmoid(report_logits).cpu().numpy()
    
    report_tpr = _tpr_at_fpr(y_report.astype(int), report_scores)
    
    model.cpu()
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_best.pt"))
    return model, {"val_tpr_at_10fpr": report_tpr}


def _predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32)).squeeze(1)).numpy()


def _save_mlp(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_best.pt"))


def _load_mlp(save_dir):
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    model.load_state_dict(torch.load(os.path.join(save_dir, "mlp_best.pt"), map_location="cpu"))
    model.eval()
    return model


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _train_xgb(X_train, y_train, groups_train, X_report, y_report, save_dir, hparams=None):
    params = {**config.XGB_PARAMS, **(hparams or {})}
    n_estimators          = params.pop("n_estimators", 3000)
    early_stopping_rounds = params.pop("early_stopping_rounds", config.XGB_EARLY_STOPPING_ROUNDS)
    
    X_fit, X_es, y_fit, y_es = _group_aware_es_split(X_train, y_train, groups_train)

    n_pos = int(y_fit.sum())
    n_neg = len(y_fit) - n_pos
    clf = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0, **params,
    )
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    
    report_scores = clf.predict_proba(X_report)[:, 1]
    tpr   = _tpr_at_fpr(y_report.astype(int), report_scores)
    multi = tpr_at_fpr_multi(y_report.astype(int), report_scores)
    print(f"  [XGB] best_iteration={clf.best_iteration}  TPR@1%={multi[0.01]:.4f}  TPR@5%={multi[0.05]:.4f}  TPR@10%={multi[0.10]:.4f}")
    
    collapsed = is_collapsed_classifier(y_report, report_scores)
    if collapsed: print("  [XGB] ⚠  WARNING: classifier may be collapsed")
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "xgb_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": clf.best_iteration, "multi_fpr": multi, "collapsed": collapsed}


def _predict_xgb(clf, X): return clf.predict_proba(X)[:, 1]
def _save_xgb(clf, save_dir): os.makedirs(save_dir, exist_ok=True); joblib.dump(clf, os.path.join(save_dir, "xgb_best.pkl"))
def _load_xgb(save_dir): return joblib.load(os.path.join(save_dir, "xgb_best.pkl"))

# ── Random Forest ─────────────────────────────────────────────────────────────

def _train_rf(X_train, y_train, groups_train, X_report, y_report, save_dir, hparams=None):
    params = {**config.RF_PARAMS, **(hparams or {})}
    n_pos  = int(y_train.sum())
    n_neg  = len(y_train) - n_pos
    
    # RF does not use Early Stopping, so it trains on the full pool.
    clf = SKLearnRF(**params, class_weight={0: 1.0, 1: n_neg / max(n_pos, 1)}, n_jobs=-1, random_state=config.SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
        clf.fit(X_train, y_train)
        
    report_scores = clf.predict_proba(X_report)[:, 1]
    tpr   = _tpr_at_fpr(y_report.astype(int), report_scores)
    multi = tpr_at_fpr_multi(y_report.astype(int), report_scores)
    collapsed = is_collapsed_classifier(y_report, report_scores)
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "rf_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "multi_fpr": multi, "collapsed": collapsed}

def _predict_rf(clf, X): return clf.predict_proba(X)[:, 1]
def _save_rf(clf, save_dir): os.makedirs(save_dir, exist_ok=True); joblib.dump(clf, os.path.join(save_dir, "rf_best.pkl"))
def _load_rf(save_dir): return joblib.load(os.path.join(save_dir, "rf_best.pkl"))

# ── LightGBM ──────────────────────────────────────────────────────────────────

def _lgbm_col_names(n_cols): return [f"f{i}" for i in range(n_cols)]

def _train_lgbm(X_train, y_train, groups_train, X_report, y_report, save_dir, hparams=None):
    params = {**config.LGBM_PARAMS, **(hparams or {})}
    early_stopping_rounds = params.pop("early_stopping_rounds", config.LGBM_EARLY_STOPPING_ROUNDS)

    if params.get("min_child_samples", 20) > 30: params["min_child_samples"] = 30
    if params.get("colsample_bytree", 0.3) < 0.2: params["colsample_bytree"] = 0.2
    params.pop("is_unbalance", None)

    X_fit, X_es, y_fit, y_es = _group_aware_es_split(X_train, y_train, groups_train)
    n_pos = int(y_fit.sum())
    n_neg = len(y_fit) - n_pos

    cols     = _lgbm_col_names(X_fit.shape[1])
    X_fit_df = pd.DataFrame(X_fit, columns=cols)
    X_es_df  = pd.DataFrame(X_es, columns=cols)
    X_rep_df = pd.DataFrame(X_report, columns=cols)

    clf = LGBMClassifier(scale_pos_weight=n_neg / max(n_pos, 1), **params)
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=-1)]
    clf.fit(X_fit_df, y_fit, eval_set=[(X_es_df, y_es)], callbacks=callbacks)

    report_scores = clf.predict_proba(X_rep_df)[:, 1]
    tpr   = _tpr_at_fpr(y_report.astype(int), report_scores)
    multi = tpr_at_fpr_multi(y_report.astype(int), report_scores)
    best_iter = clf.best_iteration_ if clf.best_iteration_ else params.get("n_estimators", 3000)

    collapsed = is_collapsed_classifier(y_report, report_scores)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "lgbm_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": best_iter, "multi_fpr": multi, "collapsed": collapsed}

def _predict_lgbm(clf, X):
    if not isinstance(X, pd.DataFrame):
        cols = _lgbm_col_names(X.shape[1])
        X    = pd.DataFrame(X, columns=cols)
    return clf.predict_proba(X)[:, 1]

def _save_lgbm(clf, save_dir): os.makedirs(save_dir, exist_ok=True); joblib.dump(clf, os.path.join(save_dir, "lgbm_best.pkl"))
def _load_lgbm(save_dir): return joblib.load(os.path.join(save_dir, "lgbm_best.pkl"))


# ── CatBoost ──────────────────────────────────────────────────────────────────

def _train_cat(X_train, y_train, groups_train, X_report, y_report, save_dir, hparams=None):
    params = {**config.CAT_PARAMS, **(hparams or {})}

    if "rsm" in params and "colsample_bylevel" in params: params.pop("colsample_bylevel")
    SUBSAMPLE_COMPATIBLE = {"Bernoulli", "Poisson"}
    bt = params.get("bootstrap_type", "Bayesian")
    if "subsample" in params and bt not in SUBSAMPLE_COMPATIBLE: params.pop("subsample")
    if "bagging_temperature" in params and bt not in ("Bayesian", "No"): params.pop("bagging_temperature")

    X_fit, X_es, y_fit, y_es = _group_aware_es_split(X_train, y_train, groups_train)
    n_pos = int(y_fit.sum())
    n_neg = len(y_fit) - n_pos

    clf = CatBoostClassifier(class_weights=[1.0, n_neg / max(n_pos, 1)], **params)
    clf.fit(X_fit, y_fit, eval_set=(X_es, y_es), verbose=False)

    report_scores = clf.predict_proba(X_report)[:, 1]
    tpr   = _tpr_at_fpr(y_report.astype(int), report_scores)
    multi = tpr_at_fpr_multi(y_report.astype(int), report_scores)
    raw_best  = clf.get_best_iteration()
    best_iter = raw_best if raw_best is not None else params.get("iterations", 3000)
    collapsed = is_collapsed_classifier(y_report, report_scores)

    os.makedirs(save_dir, exist_ok=True)
    clf.save_model(os.path.join(save_dir, "cat_best.cbm"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": best_iter, "multi_fpr": multi, "collapsed": collapsed}

def _predict_cat(clf, X): return clf.predict_proba(X)[:, 1]
def _save_cat(clf, save_dir): os.makedirs(save_dir, exist_ok=True); clf.save_model(os.path.join(save_dir, "cat_best.cbm"))
def _load_cat(save_dir): clf = CatBoostClassifier(); clf.load_model(os.path.join(save_dir, "cat_best.cbm")); return clf


class _ClassifierEntry:
    def __init__(self, train_fn, predict_fn, save_fn, load_fn):
        self.train   = train_fn
        self.predict = predict_fn
        self.save    = save_fn
        self.load    = load_fn

CLASSIFIER_REGISTRY = {
    "mlp":  _ClassifierEntry(_train_mlp,  _predict_mlp,  _save_mlp,  _load_mlp),
    "xgb":  _ClassifierEntry(_train_xgb,  _predict_xgb,  _save_xgb,  _load_xgb),
    "rf":   _ClassifierEntry(_train_rf,   _predict_rf,   _save_rf,   _load_rf),
    "lgbm": _ClassifierEntry(_train_lgbm, _predict_lgbm, _save_lgbm, _load_lgbm),
    "cat":  _ClassifierEntry(_train_cat,  _predict_cat,  _save_cat,  _load_cat),
}

def get_classifier(name):
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY.keys())}")
    return CLASSIFIER_REGISTRY[name]

def train_classifier(X_train, y_train, groups_train, X_report, y_report, epochs=None, lr=None, device=None, save_dir=None):
    return _train_mlp(X_train, y_train, groups_train, X_report, y_report, save_dir or config.CLASSIFIER_DIR)

def load_classifier(device=None):
    return _load_mlp(config.CLASSIFIER_DIR)
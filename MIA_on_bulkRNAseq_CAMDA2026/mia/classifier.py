"""Membership inference classifiers: MLP, XGBoost, RandomForest, LightGBM, CatBoost."""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    Returns True if the model looks like it's predicting the majority class.
    Checks: accuracy suspiciously close to class prior AND AUC is poor.
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


# ── Softmax ensemble weighting ────────────────────────────────────────────────
def compute_softmax_weights(aucs: dict, temperature: float = 0.05,
                             min_auc_gate: float = 0.72) -> dict:
    names  = list(aucs.keys())
    values = np.array([aucs[n] for n in names], dtype=float)
    gate   = (values >= min_auc_gate).astype(float)

    # ── FIX: if ALL classifiers are gated out, fall back to equal weights ──
    if gate.sum() == 0:
        import warnings
        warnings.warn(
            f"[ENSEMBLE] All classifiers below AUC gate={min_auc_gate:.2f}. "
            "Falling back to equal weights across all classifiers."
        )
        gate = np.ones(len(names), dtype=float)   # lift the gate entirely

    scaled   = values / temperature
    exp_v    = np.exp(scaled - scaled[gate > 0].max())
    exp_v   *= gate                                # zero out gated entries
    weights  = exp_v / exp_v.sum()

    return {n: float(w) for n, w in zip(names, weights)}

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


def _train_mlp(X_train, y_train, X_val, y_val, save_dir, hparams=None):
    hp           = hparams or {}
    epochs       = hp.get("epochs",       config.MLP_EPOCHS)
    lr           = hp.get("lr",           config.MLP_LR)
    hidden_dim   = hp.get("hidden_dim",   config.MLP_HIDDEN_DIM)
    dropout      = hp.get("dropout",      config.MLP_DROPOUT)
    weight_decay = hp.get("weight_decay", config.MLP_WEIGHT_DECAY)
    batch_size   = hp.get("batch_size",   config.MLP_BATCH_SIZE)
    device       = config.DEVICE

    # Class-weighted loss for imbalanced data
    n_pos   = int(y_train.sum())
    n_neg   = len(y_train) - n_pos
    pos_w   = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model     = MembershipMLP(input_dim=X_train.shape[1],
                               hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    train_ds     = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t      = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t      = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_metric, best_state = -1.0, None
    history = {"train_loss": [], "val_loss": [], "val_tpr_at_10fpr": []}

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
            val_logits = model(X_val_t).squeeze(1)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_scores = torch.sigmoid(val_logits).cpu().numpy()

        tpr = _tpr_at_fpr(y_val.astype(int), val_scores)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_tpr_at_10fpr"].append(tpr)

        if tpr > best_metric:
            best_metric = tpr
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) == epochs:
            print(f"  [MLP] epoch {epoch+1:4d}/{epochs}  "
                  f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"TPR@10%FPR={tpr:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.cpu()
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_best.pt"))
    print(f"  [MLP] best TPR@10%FPR = {best_metric:.4f}")
    return model, history


def _predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32)).squeeze(1)
        ).numpy()


def _save_mlp(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_best.pt"))


def _load_mlp(save_dir):
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    model.load_state_dict(torch.load(os.path.join(save_dir, "mlp_best.pt"), map_location="cpu"))
    model.eval()
    return model


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _train_xgb(X_train, y_train, X_val, y_val, save_dir, hparams=None):
    params = {**config.XGB_PARAMS, **(hparams or {})}
    n_estimators          = params.pop("n_estimators", 3000)
    early_stopping_rounds = params.pop("early_stopping_rounds",
                                       config.XGB_EARLY_STOPPING_ROUNDS)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    clf = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0, **params,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_scores = clf.predict_proba(X_val)[:, 1]
    tpr   = _tpr_at_fpr(y_val.astype(int), val_scores)
    multi = tpr_at_fpr_multi(y_val.astype(int), val_scores)
    print(f"  [XGB] best_iteration={clf.best_iteration}  "
          f"TPR@1%={multi[0.01]:.4f}  TPR@5%={multi[0.05]:.4f}  "
          f"TPR@10%={multi[0.10]:.4f}  TPR@20%={multi[0.20]:.4f}")
    collapsed = is_collapsed_classifier(y_val, val_scores)
    if collapsed:
        print(f"  [XGB] ⚠  WARNING: classifier may be collapsed (predicting majority class)")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "xgb_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": clf.best_iteration,
                 "multi_fpr": multi, "collapsed": collapsed}


def _predict_xgb(clf, X):
    return clf.predict_proba(X)[:, 1]

def _save_xgb(clf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "xgb_best.pkl"))

def _load_xgb(save_dir):
    return joblib.load(os.path.join(save_dir, "xgb_best.pkl"))


# ── Random Forest ─────────────────────────────────────────────────────────────

def _train_rf(X_train, y_train, X_val, y_val, save_dir, hparams=None):
    params = {**config.RF_PARAMS, **(hparams or {})}
    n_pos  = int(y_train.sum())
    n_neg  = len(y_train) - n_pos
    clf = SKLearnRF(**params,
                    class_weight={0: 1.0, 1: n_neg / max(n_pos, 1)},
                    n_jobs=-1, random_state=config.SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*sklearn.utils.parallel.delayed.*",
            category=UserWarning,
        )
        clf.fit(X_train, y_train)
    val_scores = clf.predict_proba(X_val)[:, 1]
    tpr   = _tpr_at_fpr(y_val.astype(int), val_scores)
    multi = tpr_at_fpr_multi(y_val.astype(int), val_scores)
    print(f"  [RF]  TPR@1%={multi[0.01]:.4f}  TPR@5%={multi[0.05]:.4f}  "
          f"TPR@10%={multi[0.10]:.4f}  TPR@20%={multi[0.20]:.4f}")
    collapsed = is_collapsed_classifier(y_val, val_scores)
    if collapsed:
        print(f"  [RF]  ⚠  WARNING: classifier may be collapsed (predicting majority class)")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "rf_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "multi_fpr": multi, "collapsed": collapsed}


def _predict_rf(clf, X):
    return clf.predict_proba(X)[:, 1]

def _save_rf(clf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "rf_best.pkl"))

def _load_rf(save_dir):
    return joblib.load(os.path.join(save_dir, "rf_best.pkl"))


# ── LightGBM ──────────────────────────────────────────────────────────────────

def _lgbm_col_names(n_cols):
    return [f"f{i}" for i in range(n_cols)]


def _train_lgbm(X_train, y_train, X_val, y_val, save_dir, hparams=None):
    params = {**config.LGBM_PARAMS, **(hparams or {})}
    n_pos  = int(y_train.sum())
    n_neg  = len(y_train) - n_pos
    early_stopping_rounds = params.pop("early_stopping_rounds",
                                       config.LGBM_EARLY_STOPPING_ROUNDS)


    if params.get("min_child_samples", 20) > 30:
        params["min_child_samples"] = 30

    # FIX: enforce colsample_bytree floor — very low values (<0.2) cause trees
    # to almost never see minority class features, collapsing to majority prior
    if params.get("colsample_bytree", 0.3) < 0.2:
        params["colsample_bytree"] = 0.2

    # FIX: is_unbalance and scale_pos_weight conflict — never use both
    params.pop("is_unbalance", None)

    cols    = _lgbm_col_names(X_train.shape[1])
    X_tr_df = pd.DataFrame(X_train, columns=cols)
    X_vl_df = pd.DataFrame(X_val,   columns=cols)

    clf = LGBMClassifier(
        scale_pos_weight=n_neg / max(n_pos, 1),
        **params,
    )
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    clf.fit(
        X_tr_df, y_train,
        eval_set=[(X_vl_df, y_val)],
        callbacks=callbacks,
    )

    val_scores = clf.predict_proba(X_vl_df)[:, 1]
    tpr   = _tpr_at_fpr(y_val.astype(int), val_scores)
    multi = tpr_at_fpr_multi(y_val.astype(int), val_scores)
    best_iter = clf.best_iteration_ if clf.best_iteration_ else params.get("n_estimators", 3000)

    # Collapse detection — critical diagnostic
    collapsed = is_collapsed_classifier(y_val, val_scores)
    print(f"  [LGBM] best_iteration={best_iter}  "
          f"TPR@1%={multi[0.01]:.4f}  TPR@5%={multi[0.05]:.4f}  "
          f"TPR@10%={multi[0.10]:.4f}  TPR@20%={multi[0.20]:.4f}"
          + ("  ⚠  COLLAPSED — will be gated from ensemble" if collapsed else ""))

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "lgbm_best.pkl"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": best_iter,
                 "multi_fpr": multi, "collapsed": collapsed}


def _predict_lgbm(clf, X):
    if not isinstance(X, pd.DataFrame):
        cols = _lgbm_col_names(X.shape[1])
        X    = pd.DataFrame(X, columns=cols)
    return clf.predict_proba(X)[:, 1]

def _save_lgbm(clf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(save_dir, "lgbm_best.pkl"))

def _load_lgbm(save_dir):
    return joblib.load(os.path.join(save_dir, "lgbm_best.pkl"))


# ── CatBoost ──────────────────────────────────────────────────────────────────

def _train_cat(X_train, y_train, X_val, y_val, save_dir, hparams=None):
    params = {**config.CAT_PARAMS, **(hparams or {})}

    if "rsm" in params and "colsample_bylevel" in params:
        params.pop("colsample_bylevel")

    SUBSAMPLE_COMPATIBLE = {"Bernoulli", "Poisson"}
    bt = params.get("bootstrap_type", "Bayesian")
    if "subsample" in params and bt not in SUBSAMPLE_COMPATIBLE:
        params.pop("subsample")
    if "bagging_temperature" in params and bt not in ("Bayesian", "No"):
        params.pop("bagging_temperature")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    clf = CatBoostClassifier(
        class_weights=[1.0, n_neg / max(n_pos, 1)],
        **params,
    )
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    val_scores = clf.predict_proba(X_val)[:, 1]
    tpr   = _tpr_at_fpr(y_val.astype(int), val_scores)
    multi = tpr_at_fpr_multi(y_val.astype(int), val_scores)
    raw_best  = clf.get_best_iteration()
    best_iter = raw_best if raw_best is not None else params.get("iterations", 3000)
    collapsed = is_collapsed_classifier(y_val, val_scores)
    print(f"  [CAT] best_iteration={best_iter}  early_stop={raw_best is not None}  "
          f"TPR@1%={multi[0.01]:.4f}  TPR@5%={multi[0.05]:.4f}  "
          f"TPR@10%={multi[0.10]:.4f}  TPR@20%={multi[0.20]:.4f}"
          + ("  ⚠  COLLAPSED" if collapsed else ""))

    os.makedirs(save_dir, exist_ok=True)
    clf.save_model(os.path.join(save_dir, "cat_best.cbm"))
    return clf, {"val_tpr_at_10fpr": tpr, "best_iteration": best_iter,
                 "multi_fpr": multi, "collapsed": collapsed}

def _predict_cat(clf, X):
    return clf.predict_proba(X)[:, 1]

def _save_cat(clf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    clf.save_model(os.path.join(save_dir, "cat_best.cbm"))

def _load_cat(save_dir):
    clf = CatBoostClassifier()
    clf.load_model(os.path.join(save_dir, "cat_best.cbm"))
    return clf


# ── Registry ──────────────────────────────────────────────────────────────────

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


# ── Legacy shims ──────────────────────────────────────────────────────────────
def train_classifier(X_train, y_train, X_val, y_val,
                     epochs=None, lr=None, device=None, save_dir=None):
    return _train_mlp(X_train, y_train, X_val, y_val, save_dir or config.CLASSIFIER_DIR)

def load_classifier(device=None):
    return _load_mlp(config.CLASSIFIER_DIR)

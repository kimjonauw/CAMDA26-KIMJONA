import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import os

from mia import config

def get_tpr_at_fpr(y_true, y_scores, target_fpr=0.10):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0

dataset = "COMBINED"  # Change to "COMBINED" as needed
feat_dir = f"/home/kimjona/CAMDA26/MIA_on_bulkRNAseq_CAMDA2026/mia_output/synth_shadow/features/{dataset}"
T_SUPERSET = config.T_SUPERSET

# Load split 5
d    = np.load(os.path.join(feat_dir, "features_split_5.npz"))
raw  = d["features"]    
y    = d["y_member"]

n_samples  = raw.shape[0]
n_t_super  = len(T_SUPERSET)
n_noise    = config.N_NOISE
reshaped   = raw.reshape(n_samples, n_t_super, n_noise)

print(f"{'t':>4}  |  {'mean_AUC':>8}  {'mean_TPR@10%':>12}  |  {'lr_CV_AUC':>10}  {'lr_CV_TPR@10%':>14}")
print("-" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ti, t_val in enumerate(T_SUPERSET):
    # --- 1. Zero-Parameter Mean Baseline ---
    mean_loss = reshaped[:, ti, :].mean(axis=1)  
    auc_raw = roc_auc_score(y, -mean_loss) 
    tpr_raw = get_tpr_at_fpr(y, -mean_loss, 0.10)
    
    # --- 2. Honest Out-Of-Fold Logistic Regression ---
    X_t = reshaped[:, ti, :]  
    lr  = LogisticRegression(max_iter=1000, C=0.1)
    
    # Predict on folds the model was NOT trained on
    oof_probs = cross_val_predict(lr, X_t, y, cv=cv, method="predict_proba")[:, 1]
    auc_lr = roc_auc_score(y, oof_probs)
    tpr_lr = get_tpr_at_fpr(y, oof_probs, 0.10)
    
    print(f"t={t_val:>3} |  {auc_raw:>8.4f}  {tpr_raw:>12.4f}  |  {auc_lr:>10.4f}  {tpr_lr:>14.4f}")
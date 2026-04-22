import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHALLENGE_DIR          = "../"
BRCA_CHALLENGE_DIR     = os.path.join(CHALLENGE_DIR, "RED_TCGA-BRCA")
COMBINED_CHALLENGE_DIR = os.path.join(CHALLENGE_DIR, "RED_TCGA-COMBINED")

NOISY_DIFFUSION_ROOT = os.path.join(PROJECT_ROOT, "..", "CAMDA25_NoisyDiffusion")
BRCA_ND_DIR          = os.path.join(NOISY_DIFFUSION_ROOT, "TCGA-BRCA")
COMBINED_ND_DIR      = os.path.join(NOISY_DIFFUSION_ROOT, "TCGA-COMBINED")

MIA_OUTPUT_DIR             = os.path.join(PROJECT_ROOT, "mia_output")
SHADOW_MODEL_DIR           = os.path.join(MIA_OUTPUT_DIR, "shadow_models")
FEATURES_DIR               = os.path.join(MIA_OUTPUT_DIR, "features")
CLASSIFIER_DIR             = os.path.join(MIA_OUTPUT_DIR, "classifiers")

DATASETS = {
    "BRCA": {
        "challenge_dir":    BRCA_CHALLENGE_DIR,
        "real_tsv":         os.path.join(BRCA_CHALLENGE_DIR, "TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv"),
        "synthetic_csv":    os.path.join(BRCA_CHALLENGE_DIR, "synthetic_data_1.csv"),
        "nd_dir":           BRCA_ND_DIR,
        "splits_yaml":      os.path.join(BRCA_ND_DIR, "TCGA-BRCA_splits.yaml"),
        "nd_synthetic_dir": os.path.join(BRCA_ND_DIR, "synthetic_data"),
        "num_classes":      5,
        "label_list":       ["BRCA.Basal", "BRCA.Normal", "BRCA.Her2", "BRCA.LumA", "BRCA.LumB"],
    },
    "COMBINED": {
        "challenge_dir":    COMBINED_CHALLENGE_DIR,
        "real_tsv":         os.path.join(COMBINED_CHALLENGE_DIR, "TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv"),
        "synthetic_csv":    os.path.join(COMBINED_CHALLENGE_DIR, "synthetic_data_1.csv"),
        "reference_tsv":    os.path.join(COMBINED_CHALLENGE_DIR, "TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv"),
        "nd_dir":           COMBINED_ND_DIR,
        "splits_yaml":      os.path.join(COMBINED_ND_DIR, "TCGA-COMBINED_splits.yaml"),
        "nd_synthetic_dir": os.path.join(COMBINED_ND_DIR, "synthetic_data"),
        "num_classes":      12,
        "label_list": [
            "TCGA-KIRC", "TCGA-PRAD", "TCGA-LIHC", "TCGA-ESCA", "TCGA-BRCA", "TCGA-OV",
            "TCGA-LUSC", "TCGA-PAAD", "TCGA-KIRP", "TCGA-LUAD", "TCGA-COAD", "TCGA-SKCM",
        ],
    },
}

# ── Diffusion model ───────────────────────────────────────────────────────────
INPUT_DIM            = 978
NUM_TIMESTEPS        = 1000
HIDDEN_DIMS          = [2048, 2048]
DROPOUT              = 0.2
TIME_EMBEDDING_DIM   = 128
LABEL_EMBEDDING_DIM  = 64
ATTN_NUM_HEADS       = 0
ATTN_NUM_TOKENS      = 64
NUM_GROUPS           = 8
BETA_SCHEDULE        = "linear"
LINEAR_BETA_START    = 0.001
LINEAR_BETA_END      = 0.02
NORM_METHOD          = "quantile"

# ── Shadow model training ─────────────────────────────────────────────────────
SHADOW_EPOCHS                   = 200
SHADOW_BATCH_SIZE               = 32
SHADOW_LR                       = 0.001
SHADOW_LR_WEIGHT_DECAY          = 0.001
SHADOW_LR_PCT_START             = 0.2
SHADOW_LR_DIV_FACTOR            = 25
SHADOW_LR_FINAL_DIV_FACTOR      = 25
SHADOW_LR_ANNEAL_STRATEGY       = "cos"
SHADOW_EARLY_STOPPING           = True
SHADOW_EARLY_STOPPING_PATIENCE  = 30
SHADOW_EARLY_STOPPING_MIN_DELTA = 0.0001
SHADOW_DP_NOISE_MULTIPLIER      = 0.00001
SHADOW_MAX_GRAD_NORM            = 1.0
DUMMY_LABEL                     = 0
UNCONDITIONAL_NUM_CLASSES       = 1

FORCE_STAGES = set()

# ── Superset: the full grid extracted once from GPU ───────────────────────────
T_SUPERSET   = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750]
N_NOISE      = 600

T_LIST       = T_SUPERSET
FEATURE_MODE = "summary"

# ── MLP defaults ──────────────────────────────────────────────────────────────
def _compute_mlp_input_dim():
    import sys
    mod = sys.modules[__name__]
    n_t = len(mod.T_LIST)
    if mod.FEATURE_MODE == "summary":
        return n_t * 6 + (n_t - 1)
    return mod.N_NOISE * n_t

MLP_INPUT_DIM    = len(T_SUPERSET) * 6 + (len(T_SUPERSET) - 1)
MLP_HIDDEN_DIM   = 200
MLP_EPOCHS       = 750
MLP_LR           = 1e-4
MLP_BATCH_SIZE   = 1024
MLP_DROPOUT      = 0.0
MLP_WEIGHT_DECAY = 0.0

# ── XGBoost defaults ──────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     3000,
    "max_depth":        4,
    "learning_rate":    0.02,
    "subsample":        0.7,
    "colsample_bytree": 0.3,
    "min_child_weight": 20,
    "gamma":            1.0,
    "reg_alpha":        1.0,
    "reg_lambda":       5.0,
    "tree_method":      "hist",
    "device":           "cuda",
}
XGB_EARLY_STOPPING_ROUNDS = 50

# ── Random Forest defaults ────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":        8,
    "max_features":     "sqrt",
    "min_samples_leaf": 20,
    "max_samples":      0.7,
}

# ── LightGBM defaults ─────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators":       3000,
    "max_depth":          6,
    "learning_rate":      0.02,
    "num_leaves":         31,
    "subsample":          0.7,
    "colsample_bytree":   0.3,
    "min_child_samples":  20,
    "reg_alpha":          1.0,
    "reg_lambda":         5.0,
    "device":             "gpu",
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}
LGBM_EARLY_STOPPING_ROUNDS = 50

# ── CatBoost defaults ─────────────────────────────────────────────────────────
CAT_PARAMS = {
    "iterations":            1500,
    "depth":                 6,
    "learning_rate":         0.02,
    "l2_leaf_reg":           5.0,
    "min_data_in_leaf":      20,
    "bootstrap_type":        "Bernoulli",
    "subsample":             0.7,
    "early_stopping_rounds": 50,
    "eval_metric":           "Logloss",
    "use_best_model":        True,
    "task_type":             "CPU",
    "thread_count":          -1,
    "random_seed":           42,
    "verbose":               0,
}

# ── Active classifiers ────────────────────────────────────────────────────────
ACTIVE_CLASSIFIERS = ["mlp", "xgb", "rf", "lgbm", "cat"]

# ── Ensemble hyperparameters ──────────────────────────────────────────────────
# Single source of truth — used by step_predict_challenge AND generate_report.
# Never hardcode 0.05 / 0.72 anywhere else in the pipeline.
ENSEMBLE_TEMPERATURE  = 0.05
ENSEMBLE_MIN_AUC_GATE = 0.55

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS  = 100
OPTUNA_TIMEOUT   = None
OPTUNA_CV_FOLDS  = 5
OPTUNA_STORAGE   = None
OPTUNA_ENABLED   = True

# ── Named profiles ────────────────────────────────────────────────────────────
ACTIVE_PROFILE = "superset"

PROFILES = {
    "superset": {
        "T_SUPERSET":           [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750],
        "N_NOISE":              300,
        "FEATURE_MODE":         "summary",
        "MLP_HIDDEN_DIM":       200,
        "MLP_DROPOUT":          0.0,
        "MLP_WEIGHT_DECAY":     0.0,
        "MLP_EPOCHS":           750,
        "MLP_LR":               1e-4,
        "ACTIVE_CLASSIFIERS":   ["xgb", "rf", "lgbm", "cat"],
        "OPTUNA_ENABLED":       True,
        "OPTUNA_N_TRIALS":      100,
        "ENSEMBLE_TEMPERATURE":  0.05,
        "ENSEMBLE_MIN_AUC_GATE": 0.72,
    },
    "fast": {
        "T_SUPERSET":           [5, 10, 20, 50, 100, 200],
        "N_NOISE":              100,
        "FEATURE_MODE":         "summary",
        "MLP_HIDDEN_DIM":       64,
        "MLP_DROPOUT":          0.0,
        "MLP_WEIGHT_DECAY":     0.0,
        "MLP_EPOCHS":           100,
        "MLP_LR":               1e-4,
        "ACTIVE_CLASSIFIERS":   ["mlp", "xgb", "rf", "lgbm", "cat"],
        "OPTUNA_ENABLED":       False,
        "OPTUNA_N_TRIALS":      10,
        "ENSEMBLE_TEMPERATURE":  0.05,
        "ENSEMBLE_MIN_AUC_GATE": 0.72,
    },
}


def apply_profile(name):
    import sys
    mod = sys.modules[__name__]
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
    for key, val in PROFILES[name].items():
        setattr(mod, key, val)
    mod.T_LIST        = list(mod.T_SUPERSET)
    mod.ACTIVE_PROFILE = name
    mod.MLP_INPUT_DIM  = mod._compute_mlp_input_dim()


# ── Split configuration ───────────────────────────────────────────────────────
SPLIT_MODE         = "custom"
NUM_CUSTOM_SPLITS  = 15
CUSTOM_SPLIT_RATIO = 0.8
CUSTOM_SPLITS_DIR  = os.path.join(MIA_OUTPUT_DIR, "splits")

SYNTH_VAL_MODEL_DIR    = os.path.join(MIA_OUTPUT_DIR, "synth_val_models")
SYNTH_VAL_FEATURES_DIR = os.path.join(MIA_OUTPUT_DIR, "synth_val_features")

SYNTH_SHADOW_OUTPUT_DIR      = os.path.join(MIA_OUTPUT_DIR, "synth_shadow")
SYNTH_SHADOW_MODEL_DIR       = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "shadow_models")
SYNTH_SHADOW_FEATURES_DIR    = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "features")
SYNTH_SHADOW_CLASSIFIER_DIR  = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "classifiers")
SYNTH_SHADOW_OPTUNA_DIR      = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "optuna")

NUM_SPLITS = 5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

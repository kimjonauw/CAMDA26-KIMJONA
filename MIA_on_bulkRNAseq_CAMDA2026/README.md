# MIA on Bulk RNA-seq (CAMDA 2026)

Membership inference attack on diffusion-generated synthetic bulk RNA-seq data
for the CAMDA 2026 Health Privacy Challenge (Red Team).

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Paths, hyperparameters, dataset configs (BRCA + COMBINED), split modes, named profiles |
| `data_utils.py` | Data loading, splits YAML, custom split generation, label prediction |
| `shadow_model.py` | Train/load EmbeddedDiffusion shadow models (QuantileTransformer + OneCycleLR + DP noise) |
| `loss_features.py` | Extract loss trajectories, `summarize_features` (per-timestep stats), `prepare_features` (dispatch on profile) |
| `classifier.py` | MembershipMLP (configurable dims, optional dropout, weight decay; best by TPR@10%FPR) |
| `attack.py` | Main pipeline: real-data shadows + synthetic validation |
| `attack_synth_shadow.py` | Alternative pipeline: synthetic-data shadows (no domain gap) |

## Two Attack Pipelines

### 1. Main pipeline (`attack.py`)

Shadows trained on **real data** subsets. Includes synthetic validation to measure
the real->synthetic domain gap at inference.

```
STEP 0 (if custom splits): Generate N random 80/20 splits of real data
STEP 1: Train N shadow models on real data subsets
STEP 2: Extract loss features for all real samples (one per shadow)
STEP 3: Train MLP classifier (70/30 train/val split of N shadows)
STEP 4: Per-split evaluation
STEP 5: Synthetic validation (train on ND synth, evaluate MLP — domain gap)
STEP 6: Challenge prediction (target proxy on challenge synthetic)
```

Split modes (configured via `SPLIT_MODE` in `config.py`):
- `"custom"`: generates random 80/20 splits (configurable N, ratio, seed)
- `"noisydiffusion"`: uses NoisyDiffusion repo's 5-fold splits YAML

```bash
python -m mia.attack --dataset BRCA --device cuda:0
python -m mia.attack --dataset BRCA --profile tuned          # use tuned profile
python -m mia.attack --dataset BRCA --force shadows,features  # force re-run of steps 1-2
python -m mia.attack --dataset BRCA --force all               # force re-run of everything
```

### 2. Synth-shadow pipeline (`attack_synth_shadow.py`)

Shadows trained on **ND synthetic data** — eliminates the domain gap between
shadow training and challenge inference (both use synthetic-trained models).

```
STEP 1: Train 5 shadow models on ND synthetic datasets
STEP 2: Extract loss features (scaler fitted on synth data)
STEP 3: Train MLP classifier (splits 1-3 train, 4-5 val)
STEP 4: Per-split evaluation
STEP 5: Challenge prediction
```

All artifacts saved under `mia_output/synth_shadow/` (separate from main pipeline).
Challenge predictions: `synthetic_data_1_predictions_synth_shadow.csv`.

```bash
python -m mia.attack_synth_shadow --dataset BRCA --device cuda:0
python -m mia.attack_synth_shadow --dataset BRCA --profile tuned --force all
```

## Config Profiles

Two named profiles switchable via `--profile baseline|tuned`:

| Parameter | `baseline` | `tuned` |
|-----------|-----------|---------|
| T_LIST | [5,10,20,30,40,50,100] | [1,2,5,10,20,50,100,200] |
| N_NOISE | 300 | 100 |
| FEATURE_MODE | "raw" (2100 dims) | "summary" (32 dims) |
| MLP_HIDDEN_DIM | 200 | 64 |
| MLP_DROPOUT | 0.0 | 0.3 |
| MLP_WEIGHT_DECAY | 0.0 | 1e-3 |
| MLP_EPOCHS | 750 | 2000 |
| MLP_LR | 1e-4 | 1e-4 |

**Summary mode** computes per-timestep mean, std, min, max across noise vectors (4 x len(T_LIST) features).
Raw features are always saved as-is; summarization happens at MLP time.
Changing profiles never requires re-running shadow training. Only re-extract features if T_LIST or N_NOISE change.

## Skip Logic

Steps 0-2 (splits, shadow training, feature extraction) and synthetic validation skip when their
outputs already exist. Use `--force <stages>` to override:

| Stage name | What it forces |
|------------|---------------|
| `shadows` | Re-train shadow models |
| `features` | Re-extract loss features |
| `synth_val` | Re-train synth validation models + features |
| `challenge` | Re-train target proxy |
| `all` | Force everything |

Comma-separated: `--force shadows,features`. MLP training and evaluation always re-run.

## Key Design Decisions

- **Real data** is a genes x samples TSV (transposed on load), not pre-split CSVs
- **Labels** inferred via KNN on NoisyDiffusion's labeled synthetic data (no separate label files)
- **Ground truth membership** from NoisyDiffusion splits YAML (train_index/test_index)
- **Shadow models** train unconditionally (dummy label=0)
- Both **BRCA** (5 subtypes, 1089 samples) and **COMBINED** (12 cancer types, ~4300 samples) supported

## Hyperparameters

- Shadow model: 200 epochs, batch 32, LR 0.001, OneCycleLR, DP noise 1e-5
- Diffusion model: EmbeddedDiffusion, hidden [2048,2048], time_emb 128, label_emb 64
- Feature extraction & MLP: see **Config Profiles** above

## Results (BRCA, real-data shadows)

- Best val TPR@10%FPR: 0.58
- Per-split: Split1=0.67, Split2=0.71, Split3=0.50, Split4=0.54, Split5=0.56
- Synthetic validation TPR@10%FPR: ~0.14 (domain gap from real->synth)

## Output Structure

```
mia_output/
├── shadow_models/BRCA/       # real-data shadow_split_{1-5}.pt, target_proxy.pt
├── features/BRCA/            # features_split_{1-5}.npz
├── classifiers/              # mlp_best.pt (main pipeline)
├── splits/BRCA/              # splits.json (custom splits)
├── synth_val_models/BRCA/    # synth_val_split_{1-5}.pt
├── synth_val_features/BRCA/  # features_synth_val_split_{1-5}.npz
└── synth_shadow/             # alternative pipeline artifacts
    ├── shadow_models/BRCA/   # synth-trained shadow_split_{1-5}.pt
    ├── features/BRCA/        # features from synth shadows
    └── classifiers/          # mlp_best.pt (synth-shadow pipeline)
```

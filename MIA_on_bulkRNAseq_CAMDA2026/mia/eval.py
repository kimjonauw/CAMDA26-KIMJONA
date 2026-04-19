import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, average_precision_score
import copy

# ==========================================
# 1. METRICS HELPER
# ==========================================
def get_tpr_at_fpr(y_true, y_prob, target_fpr=0.10):
    """Calculates TPR at a specific FPR threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find the index of the highest FPR that is <= target_fpr
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]

# ==========================================
# 2. PYTORCH TO SCIKIT-LEARN WRAPPER
# ==========================================
class SklearnPyTorchWrapper:
    """Wraps a PyTorch model so it can be used like an sklearn model."""
    def __init__(self, input_dim, epochs=300, batch_size=64, lr=1e-3, patience=20):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Regularized MLP Architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        ).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Dynamically calculate class imbalance: (count_negative / count_positive)
        # For your 2613 members vs 654 non-members, this will naturally be ~0.25
        pos_weight_val = float((y_train == 0).sum() / (y_train == 1).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(self.device))
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Optional Early Stopping using validation data
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)[:, 1]
                # We use AUPRC for early stopping as it handles imbalance better than loss
                val_auprc = average_precision_score(y_val, val_probs)
                
                # Maximizing AUPRC (invert logic for loss)
                val_loss_proxy = -val_auprc 
                if val_loss_proxy < best_loss:
                    best_loss = val_loss_proxy
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve == self.patience:
                    # print(f"MLP Early stopping at epoch {epoch}")
                    break
                    
        self.model.load_state_dict(best_model_weights)
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        # Return sklearn format: [prob_class_0, prob_class_1]
        return np.hstack((1 - probs, probs))

# ==========================================
# 3. THE MODEL REGISTRY
# ==========================================
def get_model_registry(input_dim, pos_weight_ratio):
    """
    To add a new model (like regression), just add it to this dictionary.
    """
    return {
        "Logistic Regression (L1)": LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            class_weight='balanced', # Handles imbalance naturally
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight='balanced', # Handles imbalance naturally
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            scale_pos_weight=pos_weight_ratio, # Explicit imbalance handling
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        ),
        "PyTorch MLP": SklearnPyTorchWrapper(
            input_dim=input_dim, 
            epochs=200 # Cut down from 750 because we have early stopping
        )
    }

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def run_evaluation_pipeline(num_splits=5):
    # Calculate the ratio for XGBoost based on your known dataset size
    # Non-members (654) / Members (2613) = ~0.25
    pos_weight_ratio = 654 / 2613 
    
    # Store results: results[model_name][split_idx] = TPR
    all_results = {}

    for split in range(1, num_splits + 1):
        print(f"\n--- Running Split {split} ---")
        
        # TODO: Replace with your actual data loading logic
        # data = np.load(f"features_split_{split}.npz")
        # X_train, y_train = data['X_train'], data['y_train']
        # X_val, y_val = data['X_val'], data['y_val']
        
        # DUMMY DATA FOR SCRIPT EXECUTION (Remove this block)
        X_train = np.random.randn(3267, 2100)
        y_train = np.array([1]*2613 + [0]*654)
        np.random.shuffle(y_train)
        X_val = np.random.randn(2178, 2100)
        y_val = np.array([1]*1743 + [0]*435)
        np.random.shuffle(y_val)
        # --------------------------------------------------

        # Fetch fresh models for this split to avoid data leakage
        models = get_model_registry(input_dim=X_train.shape[1], pos_weight_ratio=pos_weight_ratio)
        
        for name, model in models.items():
            if name not in all_results:
                all_results[name] = []

            # 1. Train
            if name == "PyTorch MLP":
                # Pass validation data for early stopping
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)

            # 2. Predict
            y_prob = model.predict_proba(X_val)[:, 1]

            # 3. Evaluate
            tpr_10 = get_tpr_at_fpr(y_val, y_prob, target_fpr=0.10)
            auprc = average_precision_score(y_val, y_prob)
            
            all_results[name].append(tpr_10)
            print(f"[{name}] AUPRC: {auprc:.4f} | TPR@10%FPR: {tpr_10:.4f}")

    # ==========================================
    # 5. FINAL REPORT
    # ==========================================
    print("\n==========================================")
    print("FINAL 5-FOLD CROSS VALIDATION RESULTS (TPR@10%FPR)")
    print("==========================================")
    for name, metrics in all_results.items():
        mean_tpr = np.mean(metrics)
        std_tpr = np.std(metrics)
        print(f"{name.ljust(25)}: Mean = {mean_tpr:.4f}  (± {std_tpr:.4f})")

if __name__ == "__main__":
    run_evaluation_pipeline()
from typing import Dict, Any
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from imblearn.over_sampling import SMOTE

# from imblearn.combine import SMOTETomek
from collections import Counter
import numpy as np

from model import EmbeddedDiffusion, DiffusionTrainer, generate_samples

cancer_types = [
    "TCGA-KIRC",
    "TCGA-PRAD",
    "TCGA-LIHC",
    "TCGA-ESCA",
    "TCGA-BRCA",
    "TCGA-OV",
    "TCGA-LUSC",
    "TCGA-PAAD",
    "TCGA-KIRP",
    "TCGA-LUAD",
    "TCGA-COAD",
    "TCGA-SKCM",
]

brca_types = [
    "BRCA.Basal",
    "BRCA.Normal",
    "BRCA.Her2",
    "BRCA.LumA",
    "BRCA.LumB",
]


class CancerDataset(Dataset):

    def __init__(self, X_train, y_train, label_list, norm_method="standard"):
        self.label_mapping = {c: i for i, c in enumerate(label_list)}
        self.inverse_label_mapping = {i: c for i, c in enumerate(label_list)}

        self.X_train = X_train.astype(float).values

        # Choose normalization method
        if norm_method == "standard":
            self.scaler = StandardScaler()
        elif norm_method == "quantile":
            self.scaler = QuantileTransformer(output_distribution="normal")
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)

        self.y_train = y_train.replace(self.label_mapping).values
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        features = self.X_train[idx]
        label = self.y_train[idx]
        return features, label


class EmbeddedDiffusionPipeline:
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        self.config = config
        self.split_no = split_no

        self.data_path = os.path.join(
            self.config["dir_list"]["home"], self.config["dir_list"]["data_save_dir"]
        )
        self.generator_name = self.config["generator_name"]

        config_key = f"{self.generator_name}_config"

        if config_key in config:
            self.generator_config = config[config_key]
        else:
            raise ValueError(f"No config found for generator: {self.generator_name}")

        self.dataset_name = self.config["dataset_config"]["name"]
        self.label_list = cancer_types if self.dataset_name == "TCGA-COMBINED" else brca_types

        self.subtype_col_name = self.config["dataset_config"]["subtype_col_name"]
        self.real_save_dir = os.path.join(self.data_path, self.dataset_name, "real")
        self.model_save_dir = os.path.join(
            self.config["dir_list"]["home"],
            self.config["dir_list"]["model_save_dir"],
        )
        self.model_save_path = os.path.join(self.model_save_dir, f"{self.generator_name}.pth")
        self.checkpoint_save_path = os.path.join(
            self.model_save_dir,
            f"{self.generator_name}_checkpoint.pth",
        )
        # Device configuration
        self.device = self.generator_config["device"]

        # Get progressive dimensionality if specified
        hidden_dims = self.generator_config.get("hidden_dims", None)

        # Get normalization method
        self.norm_method = self.generator_config["norm_method"]

        # Get number of groups for GroupNorm
        num_groups = self.generator_config["num_groups"]

        # Model and Optimizer Init
        self.model = EmbeddedDiffusion(
            input_dim=self.generator_config["input_dim"],
            num_classes=self.generator_config["num_classes"],
            num_timesteps=self.generator_config["num_timesteps"],
            hidden_dims=hidden_dims,
            dropout=self.generator_config["dropout"],
            attn_num_tokens=self.generator_config["attn_num_tokens"],
            attn_num_heads=self.generator_config["attn_num_heads"],
            time_embedding_dim=self.generator_config["time_embedding_dim"],
            label_embedding_dim=self.generator_config["label_embedding_dim"],
            num_groups=num_groups,
        )
        self.model = self.model.to(self.device)

        # Initialize optimizer with initial learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.generator_config["learning_rate"],
            weight_decay=self.generator_config["lr_weight_decay"],
        )

        # Get parameters for improved cosine schedule
        cosine_s = self.generator_config["cosine_s"]

        # Get differential privacy parameters
        self.dp_noise_multiplier = self.generator_config.get("dp_noise_multiplier", 0.0)
        self.max_grad_norm = self.generator_config.get("max_grad_norm", 1.0)

        self.diffusion_trainer = DiffusionTrainer(
            num_timesteps=int(self.generator_config["num_timesteps"]),
            beta_schedule=self.generator_config["beta_schedule"],
            linear_beta_start=float(self.generator_config["linear_beta_start"]),
            linear_beta_end=float(self.generator_config["linear_beta_end"]),
            cosine_s=float(cosine_s),
            power_sigma_max=float(self.generator_config["power_sigma_max"]),
            power_sigma_min=float(self.generator_config["power_sigma_min"]),
            power_rho_expo=float(self.generator_config["power_rho_expo"]),
            device=self.device,
        )
        self.criterion = torch.nn.MSELoss()

        # Early stopping parameters
        self.early_stopping = self.generator_config.get("early_stopping", False)
        self.patience = self.generator_config.get("early_stopping_patience", 10)
        self.early_stopping_min_delta = self.generator_config.get(
            "early_stopping_min_delta", 0.0001
        )

    def load_dataset(self, generate=False):
        X_train = pd.read_csv(
            os.path.join(self.real_save_dir, f"X_train_real_split_{self.split_no}.csv")
        )
        y_train = pd.read_csv(
            os.path.join(self.real_save_dir, f"y_train_real_split_{self.split_no}.csv")
        )

        # Apply SMOTE to balance classes if configured
        use_stratified = self.generator_config["stratified_sampling"]

        if use_stratified and not generate:
            print("Applying SMOTE to balance the dataset...")
            # Convert to format required by SMOTE
            X_numpy = X_train.astype(float).values
            y_numpy = y_train.replace(
                {c: i for i, c in enumerate(self.label_list)}
            ).values.flatten()

            # Print class distribution before SMOTE
            print("Class distribution before SMOTE:", Counter(y_numpy))

            # Apply SMOTE to balance the dataset
            smote_upsample_to = self.generator_config.get("smote_upsample_to", None)
            if smote_upsample_to is not None:
                sampling_strategy = {i: smote_upsample_to for i in range(len(self.label_list))}
                smote = SMOTE(sampling_strategy=sampling_strategy)
            else:
                smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X_numpy, y_numpy)
            # Print class distribution after SMOTE
            print("Class distribution after SMOTE:", Counter(y_resampled))

            # Convert back to DataFrame
            X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_train = pd.DataFrame(y_resampled, columns=y_train.columns)

        # Create custom dataset
        self.dataset = CancerDataset(
            X_train,
            y_train,
            label_list=self.label_list,
            norm_method=self.norm_method,
        )
        assert len(self.dataset.label_mapping) == self.generator_config["num_classes"]

        # Create dataloader (no need for weighted sampler since we used SMOTE)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.generator_config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

    def generate(self):
        """Generate synthetic data based on your configuration."""
        self.load_dataset(generate=True)

        self.model.load_state_dict(torch.load(self.model_save_path))

        synthetic_features = []
        synthetic_labels = []

        frequency = Counter(self.dataset.y_train[:, 0].numpy())
        for label, count in frequency.items():
            # Chunk the generation to avoid memory issues
            # Do not generate more than the original number of samples
            while count > 0:
                num_samples = min(count, 600)
                generated_samples = generate_samples(
                    model=self.model,
                    diffusion_trainer=self.diffusion_trainer,
                    num_samples=num_samples,
                    label=label,
                    input_dim=self.generator_config["input_dim"],
                    device=self.device,
                    scaler=self.dataset.scaler,
                )

                # Post-process generated samples if configured
                if self.generator_config.get("clip_outliers", False):
                    # Get the percentile boundaries from the real data
                    real_data = self.dataset.X_train.cpu().numpy()
                    lower_bound = np.percentile(real_data, 1)
                    upper_bound = np.percentile(real_data, 99)

                    # Clip generated values to those boundaries
                    generated_samples = np.clip(generated_samples, lower_bound, upper_bound)

                synthetic_features.extend(generated_samples)
                synthetic_labels.extend([label] * num_samples)
                count -= num_samples

        # Flatten batches to one array
        synthetic_features = pd.DataFrame(synthetic_features)
        synthetic_labels = pd.DataFrame(synthetic_labels)

        synthetic_labels = synthetic_labels.replace(self.dataset.inverse_label_mapping)
        return synthetic_features, synthetic_labels

    def train(self):
        """Train your generator model with differential privacy and early stopping."""
        self.load_dataset()
        # Update scheduler with actual steps per epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.generator_config["learning_rate"],
            epochs=self.generator_config["epochs"],
            steps_per_epoch=len(self.dataloader),
            pct_start=self.generator_config["lr_pct_start"],
            anneal_strategy=self.generator_config["lr_anneal_strategy"],
            div_factor=self.generator_config["lr_div_factor"],
            final_div_factor=self.generator_config["lr_final_div_factor"],
        )

        start_epoch = 0
        epochs = self.generator_config["epochs"]
        torch.autograd.set_detect_anomaly(True)

        if self.generator_config.get("wandb", False):
            import wandb

            wandb.init(
                project="CAMDA_2025",
                config=self.config,
                name=self.generator_name,
                resume="allow",
            )

        self.model.train()

        # Early stopping variables
        best_loss = float("inf")
        no_improve_epochs = 0
        best_model_state = None

        # Training loop
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            current_lr = self.optimizer.param_groups[0]["lr"]

            for batch_idx, (batch_expression_target, batch_pheno) in enumerate(self.dataloader):
                # Apply differential privacy during training
                loss = self.diffusion_trainer.train_step(
                    self.model,
                    self.optimizer,
                    batch_expression_target,
                    batch_pheno,
                    self.device,
                    dp_noise_multiplier=self.dp_noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )

                # Step the scheduler after each batch
                self.scheduler.step()

                # Accumulate loss for this batch
                total_loss += loss

                if self.generator_config.get("wandb", False):
                    wandb.log({"batch_loss": loss, "epoch": epoch, "learning_rate": current_lr})

            # Log epoch metrics
            avg_loss = total_loss / len(self.dataloader)
            if self.generator_config.get("wandb", False):
                wandb.log({"epoch_loss": avg_loss})
            print(
                f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.6f}, Learning Rate: {current_lr:.6f}"
            )

            # Early stopping logic
            if self.early_stopping:
                if avg_loss < best_loss - self.early_stopping_min_delta:
                    best_loss = avg_loss
                    no_improve_epochs = 0
                    # Save best model state
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    print(f"New best model with loss: {best_loss:.6f}")
                else:
                    no_improve_epochs += 1
                    print(
                        f"No improvement for {no_improve_epochs} epochs. Best loss: {best_loss:.6f}"
                    )

                if no_improve_epochs >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

        # Save the trained model
        if self.early_stopping and best_model_state is not None:
            # If early stopping was used, the best model was already loaded
            torch.save(self.model.state_dict(), self.model_save_path)
        else:
            # Otherwise save the final model
            torch.save(self.model.state_dict(), self.model_save_path)

        print(f"Model saved to {self.model_save_path}")

        # Finish wandb run
        if self.generator_config.get("wandb", False):
            wandb.finish()
        return {"loss": avg_loss}

    def load_from_checkpoint(self):
        """Load your model from a checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_save_path}...")
        checkpoint = torch.load(self.checkpoint_save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_synthetic_data(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.DataFrame,
        experiment_name: str = "",
    ):
        data_save_dir = os.path.join(
            self.config["dir_list"]["home"], self.config["dir_list"]["data_save_dir"]
        )

        syn_save_dir = os.path.join(
            data_save_dir, self.dataset_name, "synthetic", self.generator_name, experiment_name
        )
        os.makedirs(syn_save_dir, exist_ok=True)

        # Save synthetic features and labels
        synthetic_features.to_csv(
            os.path.join(syn_save_dir, f"synthetic_data_split_{self.split_no}.csv"), index=False
        )
        synthetic_labels.to_csv(
            os.path.join(syn_save_dir, f"synthetic_labels_split_{self.split_no}.csv"), index=False
        )

        print(f"Synthetic data saved in {syn_save_dir}.")

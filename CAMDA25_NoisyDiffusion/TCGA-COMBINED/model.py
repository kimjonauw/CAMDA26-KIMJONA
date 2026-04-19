import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest


###############################################
# Sinusoidal Position Embedding (with MLP projection)
###############################################
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Computes sinusoidal position embeddings following the approach from
    'Attention Is All You Need' paper, then projects via MLP.
    """

    def __init__(self, dim, time_embedding_dim):
        super().__init__()
        self.dim = dim
        self.time_embedding_dim = time_embedding_dim

        # Half the dim for sine and half for cosine
        half_dim = self.dim // 2
        # Create a range of frequencies in log space
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

        self.mlp = nn.Sequential(
            nn.Linear(dim, self.time_embedding_dim),
            nn.SiLU(),  # Changed from Sigmoid to SiLU for better performance
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

    def forward(self, time):
        """
        time: Tensor of shape (B,) (e.g. raw timesteps as floats)
        Returns: Tensor of shape (B, time_embedding_dim)
        """
        # Expand time for broadcasting
        emb = time.unsqueeze(-1) * self.emb.unsqueeze(0)  # (B, dim//2)
        # Calculate sine and cosine embeddings
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        # If dim is odd, pad with one extra sine feature
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros([emb.shape[0], 1])], dim=-1)
        # Project through MLP to get final embedding
        emb = self.mlp(emb)  # (B, time_embedding_dim)
        return emb


###############################################
# Residual Linear Block with Conditioning and Group norm
###############################################
class ResLinear(nn.Module):
    """
    A residual block that conditions on an extra vector (e.g. time and label info).
    It applies two linear layers (with SiLU and dropout) with the conditioning
    vector concatenated at each stage.

    Uses GroupNorm instead of BatchNorm for better stability with small batches.
    """

    def __init__(self, in_channels, cond_dim, out_channels, dropout=0.1, num_groups=8):
        super().__init__()
        self.fc1 = nn.Linear(in_channels + cond_dim, out_channels)

        # Find a valid number of groups (must be a divisor of out_channels)
        self.num_groups = self._get_valid_groups(num_groups, out_channels)

        # Replace BatchNorm with GroupNorm using valid number of groups
        self.norm1 = nn.GroupNorm(self.num_groups, out_channels)
        self.fc2 = nn.Linear(out_channels + cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(self.num_groups, out_channels)

        # Replace ReLU with SiLU (Swish)
        self.act = nn.SiLU()

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.res_proj = nn.Linear(in_channels + cond_dim, out_channels)
        self.norm3 = nn.GroupNorm(self.num_groups, out_channels)

    def _get_valid_groups(self, requested_groups, num_channels):
        """Find the largest number of groups that divides num_channels and is <= requested_groups"""
        if requested_groups >= num_channels:
            return num_channels  # One channel per group (equivalent to LayerNorm)

        # Start from requested_groups and work downward
        for groups in range(requested_groups, 0, -1):
            if num_channels % groups == 0:
                print(f"Using {groups} groups for GroupNorm")
                return groups

        return 1  # Fallback to 1 group (equivalent to InstanceNorm)

    def forward(self, x, cond):
        """
        x: Tensor of shape (B, in_channels)
        cond: Tensor of shape (B, cond_dim) — the conditioning vector
        """
        h = torch.cat([x, cond], dim=1)
        h = self.fc1(h)

        # GroupNorm expects [N, C, ...] format, so unsqueeze
        h = h.unsqueeze(-1)
        h = self.norm1(h)
        h = h.squeeze(-1)

        h = self.act(h)
        if self.dropout:
            h = self.dropout(h)

        h = torch.cat([h, cond], dim=1)
        h = self.fc2(h)

        h = h.unsqueeze(-1)
        h = self.norm2(h)
        h = h.squeeze(-1)

        h = self.act(h)
        if self.dropout:
            h = self.dropout(h)

        res = self.res_proj(torch.cat([x, cond], dim=1))
        h = h + res

        h = h.unsqueeze(-1)
        h = self.norm3(h)
        h = h.squeeze(-1)

        return h


###############################################
# Attention Block
###############################################
class AttentionBlock(nn.Module):
    """
    A self-attention block that reshapes the hidden vector into tokens,
    applies multi-head self-attention (with residual connections and LayerNorm),
    and then flattens back.
    """

    def __init__(self, hidden_dim, num_tokens=16, num_heads=8, dropout=0.1):
        super().__init__()
        # Ensure hidden_dim divides evenly into num_tokens
        assert hidden_dim % num_tokens == 0, "hidden_dim must be divisible by num_tokens"
        self.num_tokens = num_tokens
        self.token_dim = hidden_dim // num_tokens
        # Also ensure token_dim is divisible by num_heads
        assert (
            self.token_dim % num_heads == 0
        ), "token_dim (hidden//num tokens) must be divisible by num_heads"

        # LayerNorm for tokens
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.token_dim)
        # A simple feed-forward network for further processing
        self.mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim * 4),
            nn.SiLU(),  # Changed from ReLU to SiLU
            nn.Dropout(dropout),
            nn.Linear(self.token_dim * 4, self.token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, hidden_dim)
        Returns: Tensor of shape (B, hidden_dim)
        """
        B, H = x.shape  # H == hidden_dim
        # Reshape x to (B, num_tokens, token_dim)
        x_tokens = x.view(B, self.num_tokens, self.token_dim)

        # Self-attention with residual connection
        x_norm = self.norm1(x_tokens)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_tokens = x_tokens + attn_out

        # Feed-forward block with residual connection
        x_norm = self.norm2(x_tokens)
        mlp_out = self.mlp(x_norm)
        x_tokens = x_tokens + mlp_out

        # Flatten tokens back to a vector
        x_out = x_tokens.view(B, H)
        return x_out


###############################################
# Diffusion Model with Progressive Dimensionality
###############################################
class EmbeddedDiffusion(nn.Module):
    def __init__(
        self,
        input_dim=978,
        num_classes=12,  # number of labels
        num_timesteps=1000,
        hidden_dims=None,  # Progressive dimensionality
        dropout=0.1,
        attn_num_tokens=16,
        attn_num_heads=8,
        time_embedding_dim=32,
        label_embedding_dim=256,
        num_groups=8,  # Groups for GroupNorm
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        self.hidden_dims = hidden_dims

        self.time_embedding_dim = time_embedding_dim
        self.label_embedding_dim = label_embedding_dim

        self.num_classes = num_classes
        # Total conditioning dimension is the sum of time and label embedding dims
        self.cond_dim = self.time_embedding_dim + self.label_embedding_dim

        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(self.input_dim, self.time_embedding_dim)
        # Label embedding
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embedding_dim)

        # Build a series of ResLinear blocks with progressive dimensions
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        # First block: input_dim to first hidden dim
        self.blocks.append(
            ResLinear(input_dim, self.cond_dim, self.hidden_dims[0], dropout, num_groups=num_groups)
        )

        # Middle blocks with progressive dimensions
        for i in range(1, len(self.hidden_dims)):
            self.blocks.append(
                ResLinear(
                    self.hidden_dims[i - 1],
                    self.cond_dim,
                    self.hidden_dims[i],
                    dropout,
                    num_groups=num_groups,
                )
            )

        # Add attention blocks if specified
        if attn_num_heads:
            for i in range(len(self.blocks)):
                block_dim = self.hidden_dims[i]
                self.attention_blocks.append(
                    AttentionBlock(
                        block_dim,
                        num_tokens=attn_num_tokens,
                        num_heads=attn_num_heads,
                        dropout=dropout,
                    )
                )

        # Final projection to original input dimension
        self.out_layer = nn.Linear(self.hidden_dims[-1], input_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x, t, labels):
        """
        x: Tensor of shape (B, input_dim) — the (possibly noisy) data.
        t: Tensor of shape (B,) containing raw timesteps (as floats in [0, num_timesteps))
        labels: Tensor of shape (B,) or (B, 1) with label indices.
        """
        t_emb = self.time_embedding(t)  # (B, time_embedding_dim)
        # Ensure labels have shape (B,)
        if labels.dim() > 1:
            labels = labels.squeeze(1)
        label_emb = self.label_embedding(labels)  # (B, label_embedding_dim)

        # Concatenate time and label conditioning
        cond = torch.cat([t_emb, label_emb], dim=1)  # (B, cond_dim)

        # Process through blocks and attention
        for block, attn in zip_longest(self.blocks, self.attention_blocks):
            if block is not None:
                x = block(x, cond)
            if attn is not None:
                x = attn(x)

        # Apply the final projection
        out = self.out_layer(x)
        return out


###############################################
# Diffusion Trainer with Differential Privacy
###############################################
class DiffusionTrainer:
    def __init__(
        self,
        num_timesteps=1000,
        beta_schedule="cosine",
        linear_beta_start=None,
        linear_beta_end=None,
        cosine_s=None,
        power_sigma_max=None,
        power_sigma_min=None,
        power_rho_expo=None,
        device="cuda",
    ):
        """Initialize noise schedule and device with improved cosine schedule."""
        self.num_timesteps = num_timesteps
        self.device = device

        if beta_schedule == "linear":
            self.beta = torch.linspace(
                linear_beta_start,
                linear_beta_end,
                num_timesteps,
                dtype=torch.float32,
            )
        elif beta_schedule == "cosine":
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            # cosine_s controls the schedule slope
            alpha_bar = torch.cos((steps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.beta = torch.clip(self.beta, 0, 0.999)
        elif beta_schedule == "power":
            # Following Equation (5) from the paper:
            # Elucidating the Design Space of Diffusion-Based Generative Models
            rho = power_rho_expo
            sigma_max = power_sigma_max if power_sigma_max is not None else 1.0
            sigma_min = power_sigma_min if power_sigma_min is not None else 0.0

            # Compute σ for timesteps 0,...,N-1:
            steps = torch.linspace(0, 1, num_timesteps)
            sigma_vals = (
                sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            # Append σ_N = 0
            sigma_vals = torch.cat([sigma_vals, torch.tensor([0.0], dtype=torch.float32)], dim=0)
            # Derive the cumulative noise scaling factors.
            alpha_bar = 1 / (1 + sigma_vals**2)
            # Then, compute β such that βᵢ = 1 - (ᾱᵢ₊₁ / ᾱᵢ)
            beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            beta = torch.clamp(beta, 0, 0.999)
            self.beta = beta.to(device)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.beta = self.beta.to(device).float()
        self.alpha = (1 - self.beta).to(device).float()
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device).float()
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar).to(device).float()

    def get_noise_schedule(self, t, x):
        """
        For a given timestep tensor t and data x, compute the noisy version of x.
        """
        t = t.to(self.device).long()
        x = x.to(self.device).float()
        noise = torch.randn_like(x, device=self.device, dtype=torch.float32)
        # Make sure to view the scalars with the right dimensions.
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        noisy_x = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
        return noisy_x, noise

    def train_step(
        self, model, optimizer, x, labels, device, dp_noise_multiplier=0.0, max_grad_norm=1.0
    ):
        """
        Training step with differential privacy.

        Args:
            model: The diffusion model
            optimizer: The optimizer
            x: Input data
            labels: Class labels
            device: Device to run on
            dp_noise_multiplier: Amount of noise to add for differential privacy (0.0 to disable)
            max_grad_norm: Maximum gradient norm for clipping
        """
        model.train()
        x = x.to(device).float()
        labels = labels.to(device)

        # Sample a random timestep (integer between 0 and num_timesteps-1)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=device, dtype=torch.long)
        noisy_x, target_noise = self.get_noise_schedule(t, x)

        # Forward pass
        predicted_noise = model(noisy_x, t.float(), labels)
        loss = F.mse_loss(predicted_noise, target_noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if dp_noise_multiplier > 0:
            # First clip gradients to bound sensitivity
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Then add noise proportional to the gradient norm bound
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=dp_noise_multiplier * max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device,
                    )
                    param.grad += noise

        # Update weights
        optimizer.step()
        return loss.item()


###############################################
# Sampling Routine (unchanged)
###############################################
def generate_samples(
    model,
    diffusion_trainer,
    num_samples,
    input_dim,
    label=None,
    device="cuda",
    scaler=None,
):
    model.eval()
    print(f"Generating {num_samples} samples for label {label}...")
    with torch.no_grad():
        # Start from pure Gaussian noise.
        x = torch.randn(num_samples, input_dim).to(device)
        labels = torch.tensor([label] * num_samples, device=device, dtype=torch.long)
        # Reverse diffusion: loop from T-1 down to 0.
        for i in range(diffusion_trainer.num_timesteps - 1, -1, -1):
            # Create a tensor of the current timestep (raw value)
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            # Pass the raw timestep (as float) to the model.
            predicted_noise = model(x, t.float(), labels)
            # Get schedule parameters and reshape them for broadcasting.
            alpha_t = diffusion_trainer.alpha[t].view(-1, 1)
            alpha_bar_t = diffusion_trainer.alpha_bar[t].view(-1, 1)
            beta_t = diffusion_trainer.beta[t].view(-1, 1)
            # Compute the mean (denoised estimate) for the current timestep.
            x = (x - beta_t * predicted_noise / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
            # For t > 0, add a noise term (this is standard DDPM sampling).
            if i > 0:
                noise = torch.randn_like(x)
                x += noise * torch.sqrt(beta_t)
        x = x.cpu().numpy()
        if scaler is not None:
            x = scaler.inverse_transform(x)
        return x

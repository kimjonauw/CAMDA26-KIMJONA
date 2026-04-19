"""Loss-trajectory feature extraction and superset slicing."""

import numpy as np
import torch
from scipy.stats import skew as scipy_skew

from . import config


@torch.no_grad()
def extract_loss_features(
    model,
    diffusion_trainer,
    X,
    y_int,
    t_list=None,
    n_noise=None,
    device=None,
    batch_size=512,  # now = samples per batch, not noise per batch
):
    t_list  = t_list  or config.T_LIST
    n_noise = n_noise or config.N_NOISE
    device  = device  or config.DEVICE
    model.eval()

    n_samples, input_dim = X.shape
    n_t = len(t_list)

    # Fixed noise bank on GPU: (n_noise, input_dim)
    rng = torch.Generator(device="cpu").manual_seed(config.SEED)
    noise_bank = torch.randn(n_noise, input_dim, generator=rng).to(device)

    # Output: (n_samples, n_t, n_noise)
    all_losses = np.empty((n_samples, n_t, n_noise), dtype=np.float32)

    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    y_gpu = torch.tensor(y_int, dtype=torch.long, device=device)

    for ti, t_val in enumerate(t_list):
        sqrt_abar    = diffusion_trainer.sqrt_alpha_bar[t_val].item()
        sqrt_1m_abar = diffusion_trainer.sqrt_one_minus_alpha_bar[t_val].item()
        t_scalar     = torch.tensor(t_val, dtype=torch.float32, device=device)

        # Process samples in batches
        for start in range(0, n_samples, batch_size):
            end   = min(start + batch_size, n_samples)
            x0    = X_gpu[start:end]          # (B, input_dim)
            label = y_gpu[start:end]          # (B,)
            B     = x0.shape[0]

            # Expand: (B, n_noise, input_dim)
            x0_exp    = x0.unsqueeze(1).expand(-1, n_noise, -1)
            noise_exp = noise_bank.unsqueeze(0).expand(B, -1, -1)

            x_noisy = sqrt_abar * x0_exp + sqrt_1m_abar * noise_exp
            # Flatten to (B*n_noise, input_dim)
            x_flat = x_noisy.reshape(B * n_noise, input_dim)
            t_flat = t_scalar.expand(B * n_noise)
            l_flat = label.unsqueeze(1).expand(-1, n_noise).reshape(B * n_noise)

            eps_pred = model(x_flat, t_flat, l_flat)
            eps_true = noise_exp.reshape(B * n_noise, input_dim)

            mse = ((eps_pred - eps_true) ** 2).mean(dim=1)   # (B*n_noise,)
            mse = mse.reshape(B, n_noise).cpu().numpy()
            all_losses[start:end, ti, :] = mse

        print(f"  [loss features] t={t_val} done ({ti+1}/{n_t})")

    return all_losses.reshape(n_samples, n_t * n_noise)
def slice_raw_features(raw_superset, selected_t_indices, noise_budget):
    """Slice a superset raw array without touching the GPU.

    Parameters
    ----------
    raw_superset      : np.ndarray, shape (n_samples, N_T_SUPERSET * N_NOISE)
                        The full raw array stored to disk in step 2.
    selected_t_indices: list[int]
                        Indices into T_SUPERSET to keep (0-based).
                        e.g. [0,1,2,6,8] selects t=[1,2,5,50,150]
    noise_budget      : int
                        Number of noise vectors to use (1 … N_NOISE).
                        First `noise_budget` columns are kept — same fixed
                        seed, so this is a consistent subset.

    Returns
    -------
    sliced : np.ndarray, shape (n_samples, len(selected_t_indices) * noise_budget)
    """
    n_samples      = raw_superset.shape[0]
    n_t_superset   = len(config.T_SUPERSET)
    n_noise_full   = config.N_NOISE

    reshaped = raw_superset.reshape(n_samples, n_t_superset, n_noise_full)
    sliced   = reshaped[:, selected_t_indices, :][:, :, :noise_budget]
    return sliced.reshape(n_samples, len(selected_t_indices) * noise_budget)


def summarize_features(raw, n_noise, n_t):
    """Summarize raw (n_samples, n_t * n_noise) into per-timestep statistics.

    Per timestep: mean, std, min, max, median, skew   → n_t * 6
    Inter-timestep mean gradient                       → n_t - 1
    Total output dim: n_t * 6 + (n_t - 1)
    """
    n_samples = raw.shape[0]
    reshaped  = raw.reshape(n_samples, n_t, n_noise)

    mean     = reshaped.mean(axis=2)
    std      = reshaped.std(axis=2)
    mn       = reshaped.min(axis=2)
    mx       = reshaped.max(axis=2)
    median   = np.median(reshaped, axis=2)
    skewness = np.nan_to_num(scipy_skew(reshaped, axis=2), nan=0.0)
    gradient = np.diff(mean, axis=1)

    return np.concatenate([mean, std, mn, mx, median, skewness, gradient], axis=1)


def summary_input_dim(n_t):
    return n_t * 6 + (n_t - 1)


def prepare_features(raw):
    """Used for final inference only (not inside Optuna trials).
    Slices using the current config.T_LIST / config.N_NOISE, then summarizes.
    """
    # Find indices of T_LIST inside T_SUPERSET
    t_indices = [config.T_SUPERSET.index(t) for t in config.T_LIST]
    sliced    = slice_raw_features(raw, t_indices, config.N_NOISE)
    n_t       = len(config.T_LIST)
    return summarize_features(sliced, config.N_NOISE, n_t)
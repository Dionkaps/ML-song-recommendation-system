import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# Project imports (reuse existing helpers & config just like kmeans.py/gmm.py)
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv  # noqa: E402
from src.ui.modern_ui import launch_ui  # noqa: E402
from src.clustering.kmeans import (  # noqa: E402
    compute_cluster_range,
    compute_visualization_coords,
    load_clustering_dataset,
)


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
from joblib import Parallel, delayed


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_optimal_components_bic(
    X: np.ndarray,
    min_components: int = 2,
    max_components: int = 15,
    n_jobs: int = 1,
) -> Tuple[int, List[float], List[float]]:
    """
    Select optimal number of components using BIC on a GMM fit.
    
    NOTE: For VaDE, this should ideally be called on the LATENT space
    after pretraining, not on the input features. Use 
    _select_optimal_components_latent() instead when a pretrained model is available.
    
    Uses parallel processing for faster computation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        min_components: Minimum number of components to try
        max_components: Maximum number of components to try
        n_jobs: Number of parallel jobs (-1 = all cores)
        
    Returns:
        Tuple of (optimal_n_components, list_of_bic_scores, list_of_silhouette_scores)
    """
    # Cap max components based on data size to avoid numerical issues
    n_samples = X.shape[0]
    max_sensible = min(max_components, n_samples // 10, 30)
    if max_components > max_sensible:
        print(f"  Capping max components from {max_components} to {max_sensible} (based on data size)")
        max_components = max(min_components, max_sensible)
    
    sample_size = min(5000, X.shape[0])

    def evaluate_n(n: int) -> Tuple[int, float, float, bool]:
        """Evaluate latent-space BIC and silhouette for a single n value."""
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='diag',  # Match VaDE's diagonal covariance
                max_iter=100,
                n_init=3,
                random_state=42,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            labels = gmm.predict(X)
            unique_labels = np.unique(labels)
            silhouette = float("nan")
            if len(unique_labels) > 1 and len(unique_labels) < len(X):
                silhouette_kwargs = {"random_state": 42}
                if sample_size < len(X):
                    silhouette_kwargs["sample_size"] = sample_size
                silhouette = float(silhouette_score(X, labels, **silhouette_kwargs))
            return n, bic, silhouette, gmm.converged_
        except Exception:
            return n, float("inf"), float("nan"), False
    
    print(f"Searching for optimal number of components ({min_components}-{max_components})...")
    
    # Parallel evaluation
    n_values = list(range(min_components, max_components + 1))
    if n_jobs == 1:
        results = [evaluate_n(n) for n in n_values]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
            delayed(evaluate_n)(n) for n in n_values
        )
    
    # Process results
    best_n = min_components
    best_bic = float('inf')
    bic_scores: List[float] = []
    silhouette_scores: List[float] = []
    converged_rows = []

    for n, bic, silhouette, converged in sorted(results, key=lambda x: x[0]):
        bic_scores.append(bic)
        silhouette_scores.append(silhouette)
        if converged:
            converged_rows.append((n, bic, silhouette))

    if converged_rows:
        best_bic = min(row[1] for row in converged_rows)
        bic_tolerance = 10.0
        candidate_rows = [row for row in converged_rows if row[1] <= best_bic + bic_tolerance]
        best_n = min(
            candidate_rows,
            key=lambda row: (
                -np.nan_to_num(row[2], nan=-1.0),
                row[1],
                row[0],
            ),
        )[0]

    print(f"Optimal components (latent BIC) -> {best_n} (BIC={best_bic:.2f})")
    return best_n, bic_scores, silhouette_scores


def _select_optimal_components_latent(
    model: 'VaDE',
    X: np.ndarray,
    min_components: int = 2,
    max_components: int = 15,
) -> Tuple[int, List[float], List[float]]:
    """
    Select optimal number of components using BIC on the LATENT space.
    
    This is the correct approach for VaDE: run BIC on the encoded representations
    after pretraining the autoencoder, since clustering happens in latent space.
    
    Args:
        model: Pretrained VaDE model (encoder must be trained)
        X: Input feature matrix (will be encoded)
        min_components: Minimum number of components to try
        max_components: Maximum number of components to try
        
    Returns:
        Tuple of (optimal_n_components, list_of_bic_scores, list_of_silhouette_scores)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode all data to latent space
    print("Encoding data to latent space for component selection...")
    with torch.no_grad():
        B = 4096
        z_mus: List[np.ndarray] = []
        for i in range(0, X.shape[0], B):
            x = torch.from_numpy(X[i : i + B]).float().to(device)
            mu, _ = model.encode(x)
            z_mus.append(mu.cpu().numpy())
        Z = np.vstack(z_mus)
    
    print(f"Latent space shape: {Z.shape}")
    
    # Now run BIC selection on latent space
    return _select_optimal_components_bic(Z, min_components, max_components)


# -----------------------------------------------------------------------------
# VaDE model
#   Variational Deep Embedding: AE with GMM prior in latent space
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dims: List[int], last_activation: Optional[nn.Module] = None):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
            elif last_activation is not None:
                layers.append(last_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VaDE(nn.Module):
    """VaDE with diagonal-covariance GMM prior.

    Encoder produces (mu, logvar). Decoder reconstructs x.
    Cluster params (pi_logits, mu_c, logvar_c) are learnable.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        enc_hidden: Optional[List[int]] = None,
        dec_hidden: Optional[List[int]] = None,
        n_components: int = 10,
    ):
        super().__init__()
        if enc_hidden is None:
            enc_hidden = [512, 256]
        if dec_hidden is None:
            dec_hidden = [256, 512]

        # Encoder heads
        self.encoder = MLP([input_dim] + enc_hidden + [latent_dim * 2])
        # Decoder
        self.decoder = MLP([latent_dim] + dec_hidden + [input_dim])

        # GMM prior parameters (learnable)
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.pi_logits = nn.Parameter(torch.zeros(n_components))  # softmax -> pi
        self.mu_c = nn.Parameter(torch.randn(n_components, latent_dim) * 0.05)
        self.logvar_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        
        # Register log(2*pi) as buffer for proper device placement
        self.register_buffer('log_2pi', torch.log(torch.tensor(2.0 * 3.141592653589793)))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        # clamp for numerical stability
        logvar = torch.clamp(logvar, min=-12.0, max=8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Inputs are standardized real values -> linear output (Gaussian like)
        return self.decoder(z)

    def log_p_z_c(self, z: torch.Tensor) -> torch.Tensor:
        """Log p(z|c) for each component c. Returns [B, K]."""
        # (B, 1, L) vs (1, K, L)
        B = z.size(0)
        z_ = z.unsqueeze(1)  # [B,1,L]
        mu = self.mu_c.unsqueeze(0)  # [1,K,L]
        logvar = self.logvar_c.unsqueeze(0)  # [1,K,L]
        # log N(z | mu_c, var_c)
        # -0.5 * [ sum( log(2pi) + logvar + (z-mu)^2 / var ) ]
        log_norm = -0.5 * (
            torch.sum(self.log_2pi + logvar + (z_ - mu) ** 2 / torch.exp(logvar), dim=2)
        )  # [B,K]
        return log_norm

    def compute_gamma(self, z: torch.Tensor) -> torch.Tensor:
        """Posterior responsibilities gamma = p(c|x) ≈ p(c|z).
        Returns [B, K]."""
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # [K]
        log_pzc = self.log_p_z_c(z)  # [B,K]
        log_gamma = log_pi.unsqueeze(0) + log_pzc  # [B,K]
        gamma = F.softmax(log_gamma, dim=1)
        # avoid exact zeros
        return torch.clamp(gamma, min=1e-8, max=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        gamma = self.compute_gamma(z)
        return x_recon, mu, logvar, z, gamma


# -----------------------------------------------------------------------------
# Loss (VaDE ELBO components)
# -----------------------------------------------------------------------------

def vade_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    gamma: torch.Tensor,
    pi_logits: torch.Tensor,
    mu_c: torch.Tensor,
    logvar_c: torch.Tensor,
    kl_weight: float = 1.0,
    kl_c_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Compute VaDE loss for a batch.

    Inputs are standardized real-valued features -> use MSE (Gaussian) recon.
    
    Args:
        kl_weight: Weight for KL(q(z|x) || p(z|c)) term (annealed during training)
        kl_c_weight: Weight for KL(q(c|x) || p(c)) term (reduced to prevent cluster collapse)
    """
    B, D = x.size()
    K, L = mu_c.size()

    # Reconstruction loss: Gaussian with unit variance ~ MSE (sum over features)
    recon = 0.5 * torch.sum((x - x_recon) ** 2, dim=1)  # [B]

    # KL terms
    # KL(q(z|x) || p(z|c)) weighted by gamma
    # 0.5 * sum_c gamma * [ sum_j ( logvar_c - logvar - 1 + exp(logvar)/exp(logvar_c) + (mu - mu_c)^2/exp(logvar_c) ) ]
    mu_exp = mu.unsqueeze(1)       # [B,1,L]
    logvar_exp = logvar.unsqueeze(1)  # [B,1,L]

    mu_c_exp = mu_c.unsqueeze(0)          # [1,K,L]
    logvar_c_exp = logvar_c.unsqueeze(0)  # [1,K,L]

    term = (
        logvar_c_exp - logvar_exp - 1
        + torch.exp(logvar_exp - logvar_c_exp)
        + (mu_exp - mu_c_exp) ** 2 / torch.exp(logvar_c_exp)
    )  # [B,K,L]

    kl_z = 0.5 * torch.sum(gamma.unsqueeze(2) * term, dim=(1, 2))  # [B]

    # KL(q(c|x) || p(c)) - weighted down to prevent cluster collapse
    log_pi = F.log_softmax(pi_logits, dim=0)  # [K]
    kl_c = torch.sum(gamma * (torch.log(gamma + 1e-12) - log_pi.unsqueeze(0)), dim=1)  # [B]

    # Apply weights to KL terms to prevent cluster collapse
    total = torch.mean(recon + kl_weight * kl_z + kl_c_weight * kl_c)
    stats = {
        "recon": torch.mean(recon).item(),
        "kl_z": torch.mean(kl_z).item(),
        "kl_c": torch.mean(kl_c).item(),
    }
    return total, stats


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    latent_dim: int = 10
    n_components: int = 10
    enc_hidden: Optional[List[int]] = None
    dec_hidden: Optional[List[int]] = None
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    pretrain_epochs: int = 30
    train_epochs: int = 100
    device: Optional[str] = None
    # KL annealing settings to prevent cluster collapse
    kl_warmup_epochs: int = 20  # Epochs to ramp up KL weight from 0 to 1
    kl_c_weight: float = 0.1    # Weight for cluster KL term (reduced to prevent collapse)


def pretrain_autoencoder(model: VaDE, loader: DataLoader, device: torch.device, cfg: TrainConfig):
    model.train()
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pbar = tqdm(range(cfg.pretrain_epochs), desc="Pretraining autoencoder", unit="epoch")
    for epoch in pbar:
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mu, logvar = model.encode(x_batch)
            z = model.reparameterize(mu, logvar)
            x_recon = model.decode(z)
            loss = F.mse_loss(x_recon, x_batch, reduction="mean")  # simple MSE for pretrain
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(loader.dataset)
        pbar.set_postfix(MSE=f"{epoch_loss:.4f}")


def init_gmm_prior(model: VaDE, X: np.ndarray, cfg: TrainConfig):
    """Initialize (pi, mu_c, logvar_c) via sklearn GMM on encoded means."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Encode in batches to avoid OOM
        B = 4096
        z_mus: List[np.ndarray] = []
        for i in range(0, X.shape[0], B):
            x = torch.from_numpy(X[i : i + B]).float().to(device)
            mu, _ = model.encode(x)
            z_mus.append(mu.cpu().numpy())
        Z = np.vstack(z_mus)

    gmm = GaussianMixture(n_components=cfg.n_components, covariance_type="diag", random_state=42)
    gmm.fit(Z)

    with torch.no_grad():
        pi = gmm.weights_.astype(np.float32)  # [K]
        mu_c = gmm.means_.astype(np.float32)  # [K,L]
        var_c = gmm.covariances_.astype(np.float32)  # [K,L] diag

        # Clamp to avoid log(0) in edge cases
        pi_tensor = torch.clamp(torch.from_numpy(pi), min=1e-10)
        var_tensor = torch.clamp(torch.from_numpy(var_c), min=1e-10)
        
        model.pi_logits.copy_(torch.log(pi_tensor))
        model.mu_c.copy_(torch.from_numpy(mu_c))
        model.logvar_c.copy_(torch.log(var_tensor))

    print("Initialized GMM prior from encoded space (sklearn).")


def train_vade(model: VaDE, loader: DataLoader, cfg: TrainConfig):
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pbar = tqdm(range(cfg.train_epochs), desc="Training VaDE", unit="epoch")
    for epoch in pbar:
        # KL annealing: gradually increase KL weight to prevent posterior collapse
        if cfg.kl_warmup_epochs > 0:
            kl_weight = min(1.0, (epoch + 1) / cfg.kl_warmup_epochs)
        else:
            kl_weight = 1.0
            
        running = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_c": 0.0}
        n = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_recon, mu, logvar, z, gamma = model(x_batch)
            loss, stats = vade_loss(
                x_batch, x_recon, mu, logvar, gamma, model.pi_logits, model.mu_c, model.logvar_c,
                kl_weight=kl_weight,
                kl_c_weight=cfg.kl_c_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            bsz = x_batch.size(0)
            running["loss"] += loss.item() * bsz
            running["recon"] += stats["recon"] * bsz
            running["kl_z"] += stats["kl_z"] * bsz
            running["kl_c"] += stats["kl_c"] * bsz
            n += bsz

        pbar.set_postfix(
            Loss=f"{running['loss']/n:.4f}",
            Recon=f"{running['recon']/n:.4f}",
            KLz=f"{running['kl_z']/n:.4f}"
        )


# -----------------------------------------------------------------------------
# Public API — analogous to run_kmeans_clustering / run_gmm_clustering
# -----------------------------------------------------------------------------

def run_vade_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
    n_components: int = 5,
    dynamic_component_selection: bool = True,
    dynamic_min_components: Optional[int] = None,
    dynamic_max_components: Optional[int] = None,
    latent_dim: int = 10,
    enc_hidden: Optional[List[int]] = None,
    dec_hidden: Optional[List[int]] = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    pretrain_epochs: int = 20,
    train_epochs: int = 80,
    kl_warmup_epochs: int = 20,
    kl_c_weight: float = 0.1,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
):
    """
    Run VaDE (Variational Deep Embedding) clustering on audio features.
    
    VaDE combines a variational autoencoder with a Gaussian Mixture Model
    prior in the latent space, learning both feature embeddings and cluster
    assignments simultaneously.
    
    Args:
        audio_dir: Directory containing audio files organized by genre
        results_dir: Directory containing extracted feature files
        n_components: Number of GMM components (clusters) - used if dynamic_component_selection=False
        dynamic_component_selection: If True, automatically find optimal number of components using BIC
        dynamic_min_components: Minimum components to try when dynamic selection is enabled (None = auto)
        dynamic_max_components: Maximum components to try when dynamic selection is enabled (None = auto)
        latent_dim: Dimensionality of the latent space
        enc_hidden: Hidden layer sizes for encoder (default: [512, 256])
        dec_hidden: Hidden layer sizes for decoder (default: [256, 512])
        batch_size: Batch size for training
        lr: Learning rate
        pretrain_epochs: Number of epochs for autoencoder pretraining
        train_epochs: Number of epochs for full VaDE training
        kl_warmup_epochs: Epochs to ramp up KL weight from 0 to 1 (prevents posterior collapse)
        kl_c_weight: Weight for cluster KL term (reduced to prevent cluster collapse)
        n_mfcc: Number of MFCC coefficients used in features
        n_mels: Number of mel bands used in features
        include_genre: Whether genre one-hot encoding is included in features
        include_msd: Whether to include MSD metadata features (key, mode, loudness, tempo)
        songs_csv_path: Path to songs.csv containing MSD features
    
    Returns:
        Tuple of (DataFrame with results, 2D coordinates, cluster labels)
    """
    os.makedirs(results_dir, exist_ok=True)
    set_seed(42)

    # -------------------------
    # Assemble feature matrix X
    # -------------------------
    file_names, genres, unique_genres, X_prepared = load_clustering_dataset(
        audio_dir=audio_dir,
        results_dir=results_dir,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
    )

    # -------------------------
    # Compute data-driven component range
    # -------------------------
    n_samples = X_prepared.shape[0]
    genre_count_hint = len(unique_genres) if include_genre else 0
    auto_min, auto_max = compute_cluster_range(n_samples, genre_count_hint)
    min_comp = dynamic_min_components if dynamic_min_components is not None else auto_min
    max_comp = dynamic_max_components if dynamic_max_components is not None else auto_max
    
    # -------------------------
    # Create initial model for pretraining (with placeholder n_components)
    # We'll recreate with correct n_components after latent-space BIC selection
    # -------------------------
    initial_components = n_components if not dynamic_component_selection else min_comp
    
    cfg = TrainConfig(
        latent_dim=latent_dim,
        n_components=initial_components,
        enc_hidden=enc_hidden or [512, 256],
        dec_hidden=dec_hidden or [256, 512],
        batch_size=batch_size,
        lr=lr,
        pretrain_epochs=pretrain_epochs,
        train_epochs=train_epochs,
        kl_warmup_epochs=kl_warmup_epochs,
        kl_c_weight=kl_c_weight,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    
    # -------------------------
    # Dataloaders
    # -------------------------
    X_tensor = torch.from_numpy(X_prepared)
    ds = TensorDataset(X_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # -------------------------
    # Pretrain autoencoder FIRST (before component selection)
    # -------------------------
    print("\nStep 1: Pretraining autoencoder...")
    pretrain_model = VaDE(
        input_dim=X_prepared.shape[1],
        latent_dim=cfg.latent_dim,
        enc_hidden=cfg.enc_hidden,
        dec_hidden=cfg.dec_hidden,
        n_components=initial_components,  # Placeholder, will be updated
    ).to(device)
    
    pretrain_autoencoder(pretrain_model, loader, device, cfg)

    # -------------------------
    # Dynamic Component Selection (BIC-based in LATENT space)
    # -------------------------
    selected_components = n_components
    bic_scores: Optional[List[float]] = None
    silhouette_scores: Optional[List[float]] = None
    
    if dynamic_component_selection:
        print(f"\nStep 2: Selecting optimal components in LATENT space...")
        print(f"  Search range: [{min_comp}, {max_comp}] (based on {n_samples} samples)")
        
        # Run BIC on the LATENT space (after pretraining)
        selected_components, bic_scores, silhouette_scores = _select_optimal_components_latent(
            pretrain_model, X_prepared, min_comp, max_comp
        )
        print(f"Using {selected_components} components (dynamically selected via BIC in latent space)")
    else:
        print(f"\nUsing {n_components} components (fixed)")

    # -------------------------
    # Create final model with correct n_components
    # -------------------------
    print(f"\nStep 3: Creating final VaDE model with {selected_components} components...")
    cfg.n_components = selected_components
    
    model = VaDE(
        input_dim=X_prepared.shape[1],
        latent_dim=cfg.latent_dim,
        enc_hidden=cfg.enc_hidden,
        dec_hidden=cfg.dec_hidden,
        n_components=cfg.n_components,
    ).to(device)
    
    # Copy pretrained encoder/decoder weights
    model.encoder.load_state_dict(pretrain_model.encoder.state_dict())
    model.decoder.load_state_dict(pretrain_model.decoder.state_dict())
    print("  Copied pretrained encoder/decoder weights")

    # -------------------------
    # Initialize GMM prior in latent space
    # -------------------------
    print("\nStep 4: Initializing GMM prior...")
    init_gmm_prior(model, X_prepared, cfg)

    # -------------------------
    # Train full VaDE
    # -------------------------
    print("\nStep 5: Training VaDE...")
    train_vade(model, loader, cfg)

    # -------------------------
    # Inference: responsibilities & labels
    # -------------------------
    print("Computing cluster assignments...")
    model.eval()
    with torch.no_grad():
        all_gamma: List[np.ndarray] = []
        all_mu: List[np.ndarray] = []
        for (x_batch,) in DataLoader(ds, batch_size=2048, shuffle=False):
            x_batch = x_batch.to(device)
            # Use mu directly (not sampled z) for deterministic cluster assignments
            mu, logvar = model.encode(x_batch)
            gamma = model.compute_gamma(mu)  # Compute gamma from mean, not noisy sample
            all_gamma.append(gamma.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
        GAMMA = np.vstack(all_gamma)
        ZMU = np.vstack(all_mu)

    labels = GAMMA.argmax(axis=1)
    confidence = GAMMA.max(axis=1)
    
    print(
        f"VaDE formed {len(np.unique(labels))} clusters; "
        f"avg confidence: {confidence.mean():.2f}"
    )

    # Use the same prepared-feature projection as the other clustering methods
    # so downstream comparisons and plots share a common 2D basis.
    coords = compute_visualization_coords(X_prepared)

    df = pd.DataFrame(
        {
            "Song": file_names,
            "Genre": genres,
            "Cluster": labels,
            "Confidence": confidence,
            "PCA1": coords[:, 0],
            "PCA2": coords[:, 1],
            # Optional: include mean latent for debugging/analysis
            # **Do not** explode CSV size: store only first 3 dims of ZMU
            "z1": ZMU[:, 0],
            "z2": ZMU[:, 1] if ZMU.shape[1] > 1 else 0.0,
            "z3": ZMU[:, 2] if ZMU.shape[1] > 2 else 0.0,
        }
    )

    output_dir = Path("output/clustering_results")
    metrics_dir = Path("output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save BIC scores if available
    if bic_scores is not None and silhouette_scores is not None:
        selection_df = pd.DataFrame({
            "Components": list(range(min_comp, min_comp + len(bic_scores))),
            "BIC": bic_scores,
            "LatentSilhouette": silhouette_scores,
        })
        selection_path = metrics_dir / "vade_selection_criteria.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"Stored VaDE selection diagnostics -> {selection_path}")
    
    csv_path = output_dir / "audio_clustering_results_vade.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    return df, coords, labels


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DF, COORDS, LABELS = run_vade_clustering(
        audio_dir="audio_files",
        results_dir="output/features",
        n_components=10,
        dynamic_component_selection=True,
        dynamic_min_components=None,  # Auto-compute based on data
        dynamic_max_components=None,  # Auto-compute based on data
        latent_dim=10,
        pretrain_epochs=20,
        train_epochs=80,
        include_genre=fv.include_genre,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="audio_files", clustering_method="VaDE")

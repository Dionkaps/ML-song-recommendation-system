import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Project imports (reuse existing helpers & config just like kmeans.py/gmm.py)
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv  # noqa: E402
from src.ui.modern_ui import launch_ui  # noqa: E402
from src.clustering.kmeans import (  # noqa: E402
    _collect_feature_vectors,
    _load_genre_mapping,
    build_group_weights,
)


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            torch.sum(np.log(2.0 * np.pi) + logvar + (z_ - mu) ** 2 / torch.exp(logvar), dim=2)
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
) -> Tuple[torch.Tensor, dict]:
    """Compute VaDE loss for a batch.

    Inputs are standardized real-valued features -> use MSE (Gaussian) recon.
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

    # KL(q(c|x) || p(c))
    log_pi = F.log_softmax(pi_logits, dim=0)  # [K]
    kl_c = torch.sum(gamma * (torch.log(gamma + 1e-12) - log_pi.unsqueeze(0)), dim=1)  # [B]

    total = torch.mean(recon + kl_z + kl_c)
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


def pretrain_autoencoder(model: VaDE, loader: DataLoader, device: torch.device, cfg: TrainConfig):
    model.train()
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.pretrain_epochs):
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{cfg.pretrain_epochs} - MSE: {epoch_loss:.4f}")


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

        model.pi_logits.copy_(torch.log(torch.from_numpy(pi)))
        model.mu_c.copy_(torch.from_numpy(mu_c))
        model.logvar_c.copy_(torch.log(torch.from_numpy(var_c)))

    print("Initialized GMM prior from encoded space (sklearn).")


def train_vade(model: VaDE, loader: DataLoader, cfg: TrainConfig):
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.train_epochs):
        running = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_c": 0.0}
        n = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_recon, mu, logvar, z, gamma = model(x_batch)
            loss, stats = vade_loss(
                x_batch, x_recon, mu, logvar, gamma, model.pi_logits, model.mu_c, model.logvar_c
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[Train] Epoch {epoch+1}/{cfg.train_epochs} - "
                f"Loss: {running['loss']/n:.4f} | "
                f"Recon: {running['recon']/n:.4f} | "
                f"KLz: {running['kl_z']/n:.4f} | "
                f"KLc: {running['kl_c']/n:.4f}"
            )


# -----------------------------------------------------------------------------
# Public API — analogous to run_kmeans_clustering / run_gmm_clustering
# -----------------------------------------------------------------------------

def run_vade_clustering(
    audio_dir: str = "genres_original",
    results_dir: str = "output/results",
    n_components: int = 5,
    latent_dim: int = 10,
    enc_hidden: Optional[List[int]] = None,
    dec_hidden: Optional[List[int]] = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    pretrain_epochs: int = 20,
    train_epochs: int = 80,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
):
    """
    Run VaDE (Variational Deep Embedding) clustering on audio features.
    
    VaDE combines a variational autoencoder with a Gaussian Mixture Model
    prior in the latent space, learning both feature embeddings and cluster
    assignments simultaneously.
    
    Args:
        audio_dir: Directory containing audio files organized by genre
        results_dir: Directory containing extracted feature files
        n_components: Number of GMM components (clusters)
        latent_dim: Dimensionality of the latent space
        enc_hidden: Hidden layer sizes for encoder (default: [512, 256])
        dec_hidden: Hidden layer sizes for decoder (default: [256, 512])
        batch_size: Batch size for training
        lr: Learning rate
        pretrain_epochs: Number of epochs for autoencoder pretraining
        train_epochs: Number of epochs for full VaDE training
        n_mfcc: Number of MFCC coefficients used in features
        n_mels: Number of mel bands used in features
        include_genre: Whether genre one-hot encoding is included in features
    
    Returns:
        Tuple of (DataFrame with results, 2D coordinates, cluster labels)
    """
    os.makedirs(results_dir, exist_ok=True)
    set_seed(42)

    # -------------------------
    # Assemble feature matrix X
    # -------------------------
    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        results_dir, genre_map, unique_genres, include_genre
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all).astype(np.float32)

    weights = build_group_weights(
        n_mfcc=n_mfcc, 
        n_mels=n_mels, 
        n_genres=len(unique_genres),
        include_genre=include_genre
    )
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
        )
    X_weighted = (X_scaled * weights.astype(np.float32)).astype(np.float32)

    # -------------------------
    # Dataloaders
    # -------------------------
    X_tensor = torch.from_numpy(X_weighted)
    ds = TensorDataset(X_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # -------------------------
    # Model
    # -------------------------
    cfg = TrainConfig(
        latent_dim=latent_dim,
        n_components=n_components,
        enc_hidden=enc_hidden or [512, 256],
        dec_hidden=dec_hidden or [256, 512],
        batch_size=batch_size,
        lr=lr,
        pretrain_epochs=pretrain_epochs,
        train_epochs=train_epochs,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    model = VaDE(
        input_dim=X_weighted.shape[1],
        latent_dim=cfg.latent_dim,
        enc_hidden=cfg.enc_hidden,
        dec_hidden=cfg.dec_hidden,
        n_components=cfg.n_components,
    ).to(device)

    # -------------------------
    # Pretrain AE → Init GMM → Train VaDE
    # -------------------------
    print("Pretraining autoencoder...")
    pretrain_autoencoder(model, loader, device, cfg)

    print("Initializing GMM prior...")
    init_gmm_prior(model, X_weighted, cfg)

    print("Training VaDE...")
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
            x_recon, mu, logvar, z, gamma = model(x_batch)
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

    # 2D coords (for UI) using PCA on the weighted features (same as others)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_weighted)

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

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "audio_clustering_results_vade.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    return df, coords, labels


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DF, COORDS, LABELS = run_vade_clustering(
        audio_dir="genres_original",
        results_dir="output/results",
        n_components=10,
        latent_dim=10,
        pretrain_epochs=20,
        train_epochs=80,
        include_genre=fv.include_genre,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="genres_original", clustering_method="VaDE")

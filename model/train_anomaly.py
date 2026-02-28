"""
Anomaly detection model training script.
Uses an Autoencoder + Isolation Forest ensemble for robust anomaly detection.

Dataset: Synthetic IoT sensor data (temperature, vibration, pressure, current)
GCP equivalent: Vertex AI Anomaly Detection / BigQuery ML ARIMA
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

N_FEATURES = 8
N_SAMPLES = 50_000
ANOMALY_RATIO = 0.05  # 5% anomalies


# ── Autoencoder Model ────────────────────────────────────────────────── #

class AnomalyAutoencoder(nn.Module):
    """
    Reconstruction-based anomaly detector.
    High reconstruction error → likely anomaly.
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error per sample."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=-1)


# ── Data Generator ───────────────────────────────────────────────────── #

def generate_sensor_data(n_samples: int, anomaly_ratio: float, seed: int = 42):
    """Generate synthetic IoT sensor telemetry data."""
    np.random.seed(seed)
    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    # Normal operating conditions
    normal = np.column_stack([
        np.random.normal(75, 5, n_normal),   # temperature (°C)
        np.random.normal(0.2, 0.05, n_normal),  # vibration (g)
        np.random.normal(101, 2, n_normal),  # pressure (kPa)
        np.random.normal(5.0, 0.3, n_normal),   # current (A)
        np.random.normal(220, 3, n_normal),  # voltage (V)
        np.random.normal(50, 1, n_normal),   # frequency (Hz)
        np.random.normal(0.9, 0.02, n_normal),  # power factor
        np.random.normal(30, 2, n_normal),   # humidity (%)
    ])

    # Anomaly patterns (overheating, vibration spike, pressure drop)
    anomaly = np.column_stack([
        np.random.normal(110, 15, n_anomaly),   # overheating
        np.random.normal(0.8, 0.3, n_anomaly),  # vibration spike
        np.random.normal(85, 10, n_anomaly),    # pressure drop
        np.random.normal(8.0, 1.5, n_anomaly),  # current surge
        np.random.normal(200, 15, n_anomaly),   # voltage drop
        np.random.normal(48, 3, n_anomaly),     # frequency drift
        np.random.normal(0.75, 0.1, n_anomaly), # PF degradation
        np.random.normal(60, 10, n_anomaly),    # humidity spike
    ])

    X = np.vstack([normal, anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    # Shuffle
    idx = np.random.permutation(n_samples)
    df = pd.DataFrame(
        X[idx],
        columns=["temperature", "vibration", "pressure", "current",
                 "voltage", "frequency", "power_factor", "humidity"]
    )
    df["label"] = y[idx]
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    return df


# ── Training ─────────────────────────────────────────────────────────── #

def train_autoencoder(X_normal: np.ndarray, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyAutoencoder(input_dim=X_normal.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_normal, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss/len(loader):.6f}")

    return model.cpu()


def main():
    logger.info("Generating sensor data...")
    df = generate_sensor_data(N_SAMPLES, ANOMALY_RATIO)
    df.to_parquet(ARTIFACTS_DIR / "sensor_data.parquet", index=False)
    logger.info(f"Generated {len(df):,} samples ({df['label'].sum():,} anomalies)")

    feature_cols = ["temperature", "vibration", "pressure", "current",
                    "voltage", "frequency", "power_factor", "humidity"]

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[feature_cols].values)
    y_all = df["label"].values

    # Train on normal samples only
    X_normal = X_all[y_all == 0]

    # --- Autoencoder ---
    logger.info("Training autoencoder...")
    ae = train_autoencoder(X_normal, epochs=50)

    # Compute threshold on normal samples
    with torch.no_grad():
        X_t = torch.tensor(X_normal, dtype=torch.float32)
        recon_errors = ae.reconstruction_error(X_t).numpy()
    threshold = float(np.percentile(recon_errors, 95))
    logger.info(f"Reconstruction error threshold (95th pct): {threshold:.6f}")

    # --- Isolation Forest ---
    logger.info("Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=ANOMALY_RATIO, random_state=42)
    iso_forest.fit(X_normal)

    # --- Ensemble Evaluation ---
    with torch.no_grad():
        X_all_t = torch.tensor(X_all, dtype=torch.float32)
        ae_scores = ae.reconstruction_error(X_all_t).numpy()
    iso_scores = -iso_forest.decision_function(X_all)  # flip: higher = more anomalous

    # Normalize and combine
    ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    ensemble_scores = 0.5 * ae_norm + 0.5 * iso_norm

    auc = roc_auc_score(y_all, ensemble_scores)
    preds = (ensemble_scores > 0.5).astype(int)
    logger.info(f"\nEnsemble AUC-ROC: {auc:.4f}")
    logger.info(classification_report(y_all, preds, target_names=["Normal", "Anomaly"]))

    # Save artifacts
    torch.save(ae.state_dict(), ARTIFACTS_DIR / "autoencoder.pt")
    with open(ARTIFACTS_DIR / "iso_forest.pkl", "wb") as f:
        pickle.dump(iso_forest, f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "threshold": threshold,
        "auc": float(auc),
        "anomaly_ratio": ANOMALY_RATIO,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

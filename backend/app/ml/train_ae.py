from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.ae_model import Autoencoder


@dataclass
class TrainArtifacts:
    model_dir: Path
    model_version: str
    metrics: dict


def train_autoencoder(
    df: pd.DataFrame,
    feature_columns: list[str],
    artifact_root: Path,
    model_version: str,
    params: dict,
) -> TrainArtifacts:
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    model_dir = artifact_root / model_version
    model_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_columns].copy()
    missing_cols = []
    for col in feature_columns:
        if X[col].isna().any():
            ind_col = f"{col}__is_missing"
            X[ind_col] = X[col].isna().astype(float)
            missing_cols.append(ind_col)

    final_columns = feature_columns + missing_cols
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X[final_columns])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_samples = int(X_scaled.shape[0])
    if n_samples < 2:
        X_train = X_scaled
        X_val = X_scaled
    else:
        val_size = max(1, int(round(0.2 * n_samples)))
        if val_size >= n_samples:
            val_size = n_samples - 1
        X_train, X_val = train_test_split(X_scaled, test_size=val_size, random_state=params["seed"])

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    model = Autoencoder(input_dim=X_scaled.shape[1], latent_dim=params["latent_dim"], dropout=params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_left = params["patience"]

    for _epoch in range(params["epochs"]):
        model.train()
        for (xb,) in train_loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                recon = model(xb)
                val_losses.append(criterion(recon, xb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_left = params["patience"]
        else:
            # With tiny datasets we disable early stop fallback and keep training.
            if n_samples < 2:
                continue
            patience_left -= 1
            if patience_left <= 0:
                break

    if not np.isfinite(best_val):
        best_val = 0.0

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_dir / "model.pt")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(imputer, model_dir / "imputer.pkl")
    with (model_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(final_columns, f)

    metadata = {
        "epochs": params["epochs"],
        "latent_dim": params["latent_dim"],
        "lr": params["lr"],
        "batch_size": params["batch_size"],
        "dropout": params["dropout"],
        "weight_decay": params["weight_decay"],
    }
    with (model_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    metrics = {
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "val_loss": best_val,
        "input_dim": int(X_scaled.shape[1]),
    }

    return TrainArtifacts(model_dir=model_dir, model_version=model_version, metrics=metrics)

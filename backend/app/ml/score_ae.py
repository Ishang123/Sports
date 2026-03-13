from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.ml.ae_model import Autoencoder
from app.ml.explain_ae import explain_entity
from app.ml.metrics import percentile_scores
from app.models_db import EntityScore


def _prepare_model_features(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    model_df = df.copy()
    for col in model_cols:
        if col.endswith("__is_missing"):
            base_col = col.replace("__is_missing", "")
            model_df[col] = model_df[base_col].isna().astype(float) if base_col in model_df else 1.0
    for col in model_cols:
        if col not in model_df:
            model_df[col] = np.nan
    return model_df[model_cols]


def score_entities(
    session: Session,
    df: pd.DataFrame,
    artifact_dir: Path,
    model_version: str,
    window: str,
) -> pd.DataFrame:
    with (artifact_dir / "feature_columns.json").open("r", encoding="utf-8") as f:
        model_cols = json.load(f)

    scaler = joblib.load(artifact_dir / "scaler.pkl")
    imputer = joblib.load(artifact_dir / "imputer.pkl")

    model_feature_df = _prepare_model_features(df, model_cols)
    X_imp = imputer.transform(model_feature_df)
    X_scaled = scaler.transform(X_imp)

    state = torch.load(artifact_dir / "model.pt", map_location="cpu")
    latent_dim = state["encoder.3.weight"].shape[0]
    model = Autoencoder(input_dim=X_scaled.shape[1], latent_dim=latent_dim)
    model.load_state_dict(state)
    model.eval()

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model(x_tensor).numpy()

    sq_err = (X_scaled - recon) ** 2
    raw_scores = sq_err.mean(axis=1)
    cal_scores = percentile_scores(raw_scores)

    now = datetime.now(timezone.utc)
    session.execute(delete(EntityScore).where(EntityScore.window == window, EntityScore.model_version == model_version))

    out_rows = []
    base_idx = [i for i, c in enumerate(model_cols) if not c.endswith("__is_missing")]
    base_cols = [model_cols[i] for i in base_idx]
    base_df = model_feature_df[base_cols]
    for i, row in df.reset_index(drop=True).iterrows():
        row_prepared = base_df.reset_index(drop=True).iloc[i]
        exps = explain_entity(row_prepared, base_cols, sq_err[i][base_idx], base_df, top_k=3)
        out_rows.append(
            {
                "entity_id": row["entity_id"],
                "platform": row["platform"],
                "score_raw": float(raw_scores[i]),
                "anomaly_score_0_100": float(cal_scores[i]),
                "top_explanations": exps,
            }
        )
        session.add(
            EntityScore(
                entity_id=row["entity_id"],
                platform=row["platform"],
                window=window,
                model_version=model_version,
                score_raw=float(raw_scores[i]),
                anomaly_score_0_100=float(cal_scores[i]),
                top_explanations_json=json.dumps(exps),
                created_ts=now,
            )
        )

    return pd.DataFrame(out_rows)

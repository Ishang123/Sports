from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from app.db import get_session
from app.features.build_features import FEATURE_COLUMNS, build_entity_window_features
from app.ml.score_ae import score_entities
from app.ml.train_ae import train_autoencoder
from app.models_db import EntityScore


def test_feature_builder_outputs_expected_columns(seeded_db):
    with get_session() as session:
        df = build_entity_window_features(session, window="30d", as_of_ts=datetime.now(timezone.utc))
    expected = {"entity_id", "platform", *FEATURE_COLUMNS}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_train_ae_produces_artifact_files(seeded_db, tmp_path: Path):
    with get_session() as session:
        df = build_entity_window_features(session, window="30d", as_of_ts=datetime.now(timezone.utc))

    artifacts = train_autoencoder(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        artifact_root=tmp_path,
        model_version="testmodel",
        params={
            "seed": 42,
            "epochs": 3,
            "batch_size": 16,
            "latent_dim": 8,
            "lr": 1e-3,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "patience": 2,
        },
    )

    for name in ["model.pt", "scaler.pkl", "imputer.pkl", "feature_columns.json", "training_metadata.json"]:
        assert (artifacts.model_dir / name).exists()


def test_score_ae_writes_non_empty_entity_scores(seeded_db, tmp_path: Path):
    model_version = "testmodel"
    with get_session() as session:
        df = build_entity_window_features(session, window="30d", as_of_ts=datetime.now(timezone.utc))

    artifacts = train_autoencoder(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        artifact_root=tmp_path,
        model_version=model_version,
        params={
            "seed": 42,
            "epochs": 3,
            "batch_size": 16,
            "latent_dim": 8,
            "lr": 1e-3,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "patience": 2,
        },
    )

    with get_session() as session:
        scores_df = score_entities(
            session=session,
            df=df,
            artifact_dir=artifacts.model_dir,
            model_version=model_version,
            window="30d",
        )

    assert len(scores_df) > 0
    with get_session() as session:
        cnt = session.execute(select(func.count()).select_from(EntityScore)).scalar_one()
    assert cnt > 0


def test_explain_ae_returns_top_explanations_json(seeded_db, tmp_path: Path):
    model_version = "testmodel"
    with get_session() as session:
        df = build_entity_window_features(session, window="30d", as_of_ts=datetime.now(timezone.utc))

    artifacts = train_autoencoder(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        artifact_root=tmp_path,
        model_version=model_version,
        params={
            "seed": 42,
            "epochs": 3,
            "batch_size": 16,
            "latent_dim": 8,
            "lr": 1e-3,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "patience": 2,
        },
    )

    with get_session() as session:
        scored = score_entities(
            session=session,
            df=df,
            artifact_dir=artifacts.model_dir,
            model_version=model_version,
            window="30d",
        )

    exps = scored.iloc[0]["top_explanations"]
    assert isinstance(exps, list)
    assert len(exps) == 3
    assert "high reconstruction error" in exps[0]

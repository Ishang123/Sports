from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_of_value(series: pd.Series, value: float) -> float:
    if len(series) == 0:
        return 0.0
    return float((series <= value).mean() * 100.0)


def explain_entity(
    entity_row: pd.Series,
    feature_columns: list[str],
    feature_error: np.ndarray,
    all_features_df: pd.DataFrame,
    top_k: int = 3,
) -> list[str]:
    contributions = feature_error / (np.sum(feature_error) + 1e-12)
    order = np.argsort(contributions)[::-1][:top_k]

    explanations = []
    for idx in order:
        col = feature_columns[idx]
        val = entity_row[col]
        pct = percentile_of_value(all_features_df[col].fillna(all_features_df[col].median()), float(val))
        msg = f"{col}: {val:.4f} ({pct:.1f}th pct); high reconstruction error"
        explanations.append(msg)

    return explanations

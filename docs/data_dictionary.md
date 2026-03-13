# Data Dictionary

## Core tables
- `markets`: normalized market metadata across platforms.
- `trades`: normalized execution data.
- `entities`: wallet/account level identity abstraction.
- `entity_window_features`: flattened numerical feature vectors per entity/window.
- `entity_scores`: per-entity anomaly scoring outputs.
- `model_registry`: model versions and training metrics.

## Key features
- Activity: trade counts, market counts, sizing moments.
- Timing: close-proximity fractions, burstiness, inter-trade cadence.
- Concentration: market volume concentration and HHI.
- Directionality: side entropy and imbalance.
- Performance (when outcomes exist): resolved count and win-rate metrics.

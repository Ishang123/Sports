from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import get_session, init_db, resolve_db_path
from app.ml.backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="kalshi")
    parser.add_argument("--window", default="30d", choices=["7d", "30d", "90d"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--target-bets", type=int, default=5000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(ROOT / "backend" / "artifacts" / "backtest_latest.json"),
        help="Path to write JSON backtest report",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    init_db()
    print(f"Backtest using SQLite DB at: {resolve_db_path()}")

    with get_session() as session:
        result = run_backtest(
            session=session,
            platform=args.platform,
            window=args.window,
            top_n=args.top_n,
            target_bets=args.target_bets,
            cv_folds=args.cv_folds,
            random_trials=args.random_trials,
            seed=args.seed,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": result.metrics, "top_accounts": result.top_accounts}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote backtest report to: {out_path}")


if __name__ == "__main__":
    main()

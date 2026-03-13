"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { apiGet } from "../components/api";

type Row = {
  entity_id: string;
  platform: string;
  anomaly_score_0_100: number;
  market: string;
  current_price: number | null;
  current_american_odds: number | null;
  kalshi_price: number | null;
  kalshi_american_odds: number | null;
  kalshi_market: string | null;
  quarter_kelly_fraction: number | null;
  quarter_kelly_stake_usd: number | null;
  top_explanations: string[];
  num_trades: number;
  total_notional_usd: number;
  first_seen_ts: string;
  last_seen_ts: string;
};

type BacktestPayload = {
  metrics?: {
    strategy?: {
      cumulative_roi?: number;
      sharpe_ratio?: number;
      max_drawdown?: number;
      sample_size_bets?: number;
    };
    out_of_sample_validation?: {
      available?: boolean;
      cumulative_roi?: number;
      sharpe_ratio?: number;
      max_drawdown?: number;
      sample_size_bets?: number;
    };
    cross_validation?: {
      available?: boolean;
      folds_used?: number;
      mean_cumulative_roi?: number;
      mean_sharpe_ratio?: number;
      mean_sample_size_bets?: number;
    };
    error?: string;
  };
};

export default function LeaderboardPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [backtest, setBacktest] = useState<BacktestPayload | null>(null);
  const [window, setWindow] = useState<string>("30d");
  const [minTrades, setMinTrades] = useState<number>(10);
  const [marketFilter, setMarketFilter] = useState<string>("all");
  const [kalshiEquivalentOnly, setKalshiEquivalentOnly] = useState<boolean>(false);
  const [bankrollUsd, setBankrollUsd] = useState<number>(500);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const fetchRows = (): void => {
      const qs = new URLSearchParams({
        window,
        min_trades: String(minTrades),
        sort: "score_desc",
        market_filter: marketFilter,
        kalshi_equivalent_only: String(kalshiEquivalentOnly),
        bankroll_usd: String(bankrollUsd),
      });
      qs.set("platform", "kalshi");
      apiGet<Row[]>(`/api/entities?${qs.toString()}`)
        .then((data) => {
          setRows(data);
          setError("");
        })
        .catch((e) => setError(e.message));
    };

    fetchRows();
    const timer = setInterval(fetchRows, 5000);
    return () => clearInterval(timer);
  }, [window, minTrades, marketFilter, kalshiEquivalentOnly, bankrollUsd]);

  useEffect(() => {
    apiGet<BacktestPayload>("/api/backtest/latest")
      .then(setBacktest)
      .catch(() => setBacktest(null));
  }, []);

  return (
    <main>
      <h1>Prediction Market Integrity Dashboard</h1>
      <div className="card">
        <h3>Backtest Metrics</h3>
        {backtest?.metrics?.error ? (
          <p style={{ color: "var(--muted)" }}>Backtest status: {backtest.metrics.error}</p>
        ) : (
          <div className="kv-grid">
            <div>
              <strong>ROI</strong>
              <div>{backtest?.metrics?.strategy?.cumulative_roi !== undefined ? `${(backtest.metrics.strategy.cumulative_roi * 100).toFixed(2)}%` : "-"}</div>
            </div>
            <div>
              <strong>Sharpe</strong>
              <div>{backtest?.metrics?.strategy?.sharpe_ratio !== undefined ? backtest.metrics.strategy.sharpe_ratio.toFixed(3) : "-"}</div>
            </div>
            <div>
              <strong>Max Drawdown</strong>
              <div>{backtest?.metrics?.strategy?.max_drawdown !== undefined ? `$${backtest.metrics.strategy.max_drawdown.toFixed(2)}` : "-"}</div>
            </div>
            <div>
              <strong>Sample Size</strong>
              <div>{backtest?.metrics?.strategy?.sample_size_bets !== undefined ? `${Math.round(backtest.metrics.strategy.sample_size_bets)} bets` : "-"}</div>
            </div>
            <div>
              <strong>Out-of-sample validation</strong>
              <div>{backtest?.metrics?.out_of_sample_validation?.available ? "available" : "not available"}</div>
            </div>
            <div>
              <strong>Cross-validation</strong>
              <div>{backtest?.metrics?.cross_validation?.available ? `${backtest.metrics.cross_validation.folds_used ?? 0} folds` : "not available"}</div>
            </div>
            <div>
              <strong>CV Mean ROI</strong>
              <div>{backtest?.metrics?.cross_validation?.mean_cumulative_roi !== undefined ? `${(backtest.metrics.cross_validation.mean_cumulative_roi * 100).toFixed(2)}%` : "-"}</div>
            </div>
            <div>
              <strong>CV Mean Sharpe</strong>
              <div>{backtest?.metrics?.cross_validation?.mean_sharpe_ratio !== undefined ? backtest.metrics.cross_validation.mean_sharpe_ratio.toFixed(3) : "-"}</div>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Leaderboard</h3>
        <div className="filters">
          <select value={window} onChange={(e) => setWindow(e.target.value)}>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
            <option value="90d">90d</option>
          </select>
          <input
            type="number"
            min={1}
            value={minTrades}
            onChange={(e) => setMinTrades(Number(e.target.value) || 1)}
          />
          <input
            type="number"
            min={1}
            value={bankrollUsd}
            onChange={(e) => setBankrollUsd(Number(e.target.value) || 500)}
            placeholder="bankroll (usd)"
          />
          <select value={marketFilter} onChange={(e) => setMarketFilter(e.target.value)}>
            <option value="all">all markets</option>
            <option value="non_sports">non-sports</option>
            <option value="sports">sports</option>
          </select>
          <select
            value={kalshiEquivalentOnly ? "only_equiv" : "all_rows"}
            onChange={(e) => setKalshiEquivalentOnly(e.target.value === "only_equiv")}
          >
            <option value="all_rows">all rows</option>
            <option value="only_equiv">kalshi equivalent only</option>
          </select>
        </div>
        {error ? <p style={{ color: "var(--danger)" }}>{error}</p> : null}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Entity</th>
                <th>Market</th>
                <th>Current Odds</th>
                <th>Kalshi Odds</th>
                <th>Q-Kelly %</th>
                <th>Q-Kelly Stake</th>
                <th>Score</th>
                <th>Reasons</th>
                <th>Trades</th>
                <th>Volume (USD)</th>
                <th>Last Seen</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => (
                <tr key={`${r.platform}-${r.entity_id}-${idx}`}>
                  <td>
                    <Link href={`/entity/${encodeURIComponent(r.entity_id)}?window=${window}`}>
                      {r.entity_id}
                    </Link>
                    <div>{r.platform}</div>
                  </td>
                  <td>{r.market}</td>
                  <td>
                    {r.current_american_odds === null
                      ? "-"
                      : r.current_american_odds > 0
                        ? `+${r.current_american_odds}`
                        : `${r.current_american_odds}`}
                  </td>
                  <td>{r.kalshi_american_odds === null ? "-" : (r.kalshi_american_odds > 0 ? `+${r.kalshi_american_odds}` : `${r.kalshi_american_odds}`)}</td>
                  <td>{r.quarter_kelly_fraction === null ? "-" : `${(r.quarter_kelly_fraction * 100).toFixed(2)}%`}</td>
                  <td>{r.quarter_kelly_stake_usd === null ? "-" : `$${r.quarter_kelly_stake_usd.toFixed(2)}`}</td>
                  <td>
                    <span className="score-pill">{r.anomaly_score_0_100.toFixed(1)}</span>
                  </td>
                  <td>{r.top_explanations.slice(0, 3).join(" | ")}</td>
                  <td>{r.num_trades}</td>
                  <td>{r.total_notional_usd.toFixed(2)}</td>
                  <td>{new Date(r.last_seen_ts).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}

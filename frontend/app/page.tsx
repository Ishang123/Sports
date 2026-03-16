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

function formatOdds(value: number | null): string {
  if (value === null) {
    return "-";
  }
  return value > 0 ? `+${value}` : `${value}`;
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "-";
  }
  return `$${value.toFixed(2)}`;
}

export default function LeaderboardPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [backtest, setBacktest] = useState<BacktestPayload | null>(null);
  const [windowValue, setWindowValue] = useState<string>("30d");
  const [minTrades, setMinTrades] = useState<number>(10);
  const [marketFilter, setMarketFilter] = useState<string>("all");
  const [kalshiEquivalentOnly, setKalshiEquivalentOnly] = useState<boolean>(false);
  const [bankrollUsd, setBankrollUsd] = useState<number>(500);
  const [error, setError] = useState<string>("");
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const stored = window.localStorage.getItem("pmid-theme");
    const nextTheme = stored === "dark" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.dataset.theme = nextTheme;
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("pmid-theme", theme);
  }, [theme]);

  useEffect(() => {
    const fetchRows = (): void => {
      const qs = new URLSearchParams({
        window: windowValue,
        min_trades: String(minTrades),
        sort: "score_desc",
        market_filter: marketFilter,
        kalshi_equivalent_only: String(kalshiEquivalentOnly),
        bankroll_usd: String(bankrollUsd),
        platform: "kalshi",
      });
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
  }, [windowValue, minTrades, marketFilter, kalshiEquivalentOnly, bankrollUsd]);

  useEffect(() => {
    apiGet<BacktestPayload>("/api/backtest/latest")
      .then(setBacktest)
      .catch(() => setBacktest(null));
  }, []);

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Kalshi Flow Monitor</p>
          <h1>Prediction Market Integrity Dashboard</h1>
          <p className="lede">
            Live market surveillance with backtest context, anomaly ranking, and stake sizing in one place.
          </p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme((curr) => (curr === "light" ? "dark" : "light"))}
        >
          {theme === "light" ? "Dark Mode" : "Light Mode"}
        </button>
      </section>

      <section className="card metrics-card">
        <div className="section-head">
          <h3>Backtest Metrics</h3>
          <span className="section-tag">latest model</span>
        </div>
        {backtest?.metrics?.error ? (
          <p className="error-copy">Backtest status: {backtest.metrics.error}</p>
        ) : (
          <div className="kv-grid">
            <div>
              <strong>ROI</strong>
              <div>
                {backtest?.metrics?.strategy?.cumulative_roi !== undefined
                  ? `${(backtest.metrics.strategy.cumulative_roi * 100).toFixed(2)}%`
                  : "-"}
              </div>
            </div>
            <div>
              <strong>Sharpe</strong>
              <div>
                {backtest?.metrics?.strategy?.sharpe_ratio !== undefined
                  ? backtest.metrics.strategy.sharpe_ratio.toFixed(3)
                  : "-"}
              </div>
            </div>
            <div>
              <strong>Max Drawdown</strong>
              <div>
                {backtest?.metrics?.strategy?.max_drawdown !== undefined
                  ? `$${backtest.metrics.strategy.max_drawdown.toFixed(2)}`
                  : "-"}
              </div>
            </div>
            <div>
              <strong>Sample Size</strong>
              <div>
                {backtest?.metrics?.strategy?.sample_size_bets !== undefined
                  ? `${Math.round(backtest.metrics.strategy.sample_size_bets)} bets`
                  : "-"}
              </div>
            </div>
            <div>
              <strong>Out-of-sample</strong>
              <div>{backtest?.metrics?.out_of_sample_validation?.available ? "available" : "not available"}</div>
            </div>
            <div>
              <strong>Cross-validation</strong>
              <div>
                {backtest?.metrics?.cross_validation?.available
                  ? `${backtest.metrics.cross_validation.folds_used ?? 0} folds`
                  : "not available"}
              </div>
            </div>
            <div>
              <strong>CV Mean ROI</strong>
              <div>
                {backtest?.metrics?.cross_validation?.mean_cumulative_roi !== undefined
                  ? `${(backtest.metrics.cross_validation.mean_cumulative_roi * 100).toFixed(2)}%`
                  : "-"}
              </div>
            </div>
            <div>
              <strong>CV Mean Sharpe</strong>
              <div>
                {backtest?.metrics?.cross_validation?.mean_sharpe_ratio !== undefined
                  ? backtest.metrics.cross_validation.mean_sharpe_ratio.toFixed(3)
                  : "-"}
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="section-head">
          <h3>Leaderboard</h3>
          <span className="section-tag">live refresh 5s</span>
        </div>

        <div className="filters">
          <select value={windowValue} onChange={(e) => setWindowValue(e.target.value)}>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
            <option value="90d">90d</option>
          </select>
          <input
            type="number"
            min={1}
            value={minTrades}
            onChange={(e) => setMinTrades(Number(e.target.value) || 1)}
            placeholder="min trades"
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

        {error ? <p className="error-copy">{error}</p> : null}

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
              {rows.map((row, idx) => (
                <tr key={`${row.platform}-${row.entity_id}-${idx}`}>
                  <td>
                    <Link href={`/entity/${encodeURIComponent(row.entity_id)}?window=${windowValue}`}>
                      {row.entity_id}
                    </Link>
                    <div className="entity-meta">{row.platform}</div>
                  </td>
                  <td className="market-copy">{row.market}</td>
                  <td>{formatOdds(row.current_american_odds)}</td>
                  <td>{formatOdds(row.kalshi_american_odds)}</td>
                  <td>{formatPercent(row.quarter_kelly_fraction)}</td>
                  <td>{formatCurrency(row.quarter_kelly_stake_usd)}</td>
                  <td>
                    <span className="score-pill">{row.anomaly_score_0_100.toFixed(1)}</span>
                  </td>
                  <td className="reasons-copy">{row.top_explanations.slice(0, 3).join(" | ")}</td>
                  <td>{row.num_trades}</td>
                  <td>{row.total_notional_usd.toFixed(2)}</td>
                  <td>{new Date(row.last_seen_ts).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

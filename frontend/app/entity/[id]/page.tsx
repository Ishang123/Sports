"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useMemo } from "react";
import { useEffect, useState } from "react";
import { apiGet } from "../../../components/api";

type Detail = {
  entity_id: string;
  platform: string;
  window: string;
  anomaly_score_0_100: number;
  score_raw: number;
  top_explanations: string[];
  feature_snapshot: Record<string, number | null>;
  recent_trades: Array<Record<string, string | number>>;
  top_markets: Array<{ market_id: string; notional_usd: number }>;
};

function toCsv(rows: Array<Record<string, unknown>>): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => JSON.stringify(row[h] ?? "")).join(","));
  }
  return lines.join("\n");
}

export default function EntityPage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string>("");
  const windowKey = searchParams.get("window") || "30d";
  const entityId = Array.isArray(params.id) ? params.id[0] : params.id;

  useEffect(() => {
    if (!entityId) return;
    apiGet<Detail>(`/api/entities/${encodeURIComponent(entityId)}?window=${windowKey}`)
      .then(setDetail)
      .catch((e) => setError(e.message));
  }, [entityId, windowKey]);

  const snapshotRows = useMemo(() => {
    if (!detail) return [];
    return Object.entries(detail.feature_snapshot).map(([k, v]) => ({ feature: k, value: v }));
  }, [detail]);

  function exportJson(): void {
    if (!detail) return;
    const blob = new Blob([JSON.stringify(detail, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${detail.entity_id}_${windowKey}_integrity_report.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportCsv(): void {
    if (!detail) return;
    const csv = toCsv(detail.recent_trades);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${detail.entity_id}_${windowKey}_recent_trades.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (error) {
    return <p style={{ color: "var(--danger)" }}>{error}</p>;
  }
  if (!detail) {
    return <p>Loading...</p>;
  }

  return (
    <main>
      <h1>Entity: {detail.entity_id}</h1>
      <p>
        <Link href="/">Back to leaderboard</Link>
      </p>

      <div className="card">
        <h3>Integrity Risk Summary</h3>
        <p>
          Score: <span className="score-pill">{detail.anomaly_score_0_100.toFixed(1)}</span>
        </p>
        <ul>
          {detail.top_explanations.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
        <button onClick={exportJson}>Export JSON Report</button>
        <button onClick={exportCsv} style={{ marginLeft: 8 }}>
          Export CSV Trades
        </button>
      </div>

      <div className="card">
        <h3>Feature Snapshot</h3>
        <div className="kv-grid">
          {snapshotRows.map((r) => (
            <div key={r.feature}>
              <strong>{r.feature}</strong>
              <div>{String(r.value)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Top Markets</h3>
        <ul>
          {detail.top_markets.map((m) => (
            <li key={`${m.market_id}-${m.notional_usd}`}>
              {m.market_id}: {m.notional_usd.toFixed(2)}
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3>Recent Trades</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Trade</th>
                <th>Market</th>
                <th>Timestamp</th>
                <th>Side</th>
                <th>Price</th>
                <th>Qty</th>
                <th>Notional</th>
              </tr>
            </thead>
            <tbody>
              {detail.recent_trades.map((t) => (
                <tr key={String(t.trade_id)}>
                  <td>{String(t.trade_id)}</td>
                  <td>{String(t.market_id)}</td>
                  <td>{new Date(String(t.ts)).toLocaleString()}</td>
                  <td>{String(t.side)}</td>
                  <td>{Number(t.price).toFixed(4)}</td>
                  <td>{Number(t.quantity).toFixed(2)}</td>
                  <td>{Number(t.notional_usd).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}

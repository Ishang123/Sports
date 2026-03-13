"use client";

import { useParams } from "next/navigation";

export default function MarketPage() {
  const params = useParams<{ id: string }>();
  const marketId = Array.isArray(params.id) ? params.id[0] : params.id;
  return (
    <main>
      <h1>Market Detail (Optional)</h1>
      <p>Market ID: {marketId}</p>
      <p>This MVP focuses on entity-level integrity monitoring.</p>
    </main>
  );
}

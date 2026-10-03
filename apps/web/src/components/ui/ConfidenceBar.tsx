/**
 * The bar shows calibrated probability, not raw similarity, and marks both
 * thresholds. Showing the accept/review boundaries is the point: an operator
 * needs to see how close a decision was, not just which side it fell on.
 */
export function ConfidenceBar({
  confidence,
  tauHigh = 0.9,
  tauLow = 0.2,
}: {
  confidence: number;
  tauHigh?: number;
  tauLow?: number;
}) {
  const pct = Math.round(confidence * 1000) / 10;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="label">Calibrated confidence</span>
        <span className="font-mono text-sm">{pct.toFixed(1)}%</span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink">
        <div
          className="h-full rounded-full bg-accent shadow-[0_0_10px_rgba(0,240,255,0.8)] transition-[width] duration-500"
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
        <Marker at={tauLow} title="reject below" />
        <Marker at={tauHigh} title="auto-accept above" />
      </div>
      <div className="flex justify-between font-mono text-[10px] text-paper/35">
        <span>reject &lt; {tauLow.toFixed(2)}</span>
        <span>accept ≥ {tauHigh.toFixed(2)}</span>
      </div>
    </div>
  );
}

function Marker({ at, title }: { at: number; title: string }) {
  return (
    <span
      title={title}
      className="absolute top-0 h-full w-px bg-paper/50"
      style={{ left: `${at * 100}%` }}
    />
  );
}

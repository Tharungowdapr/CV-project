"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { postMatch, ApiError } from "@/lib/api";
import type { MatchResponse } from "@/types";
import { ConfidenceBar } from "@/components/ui/ConfidenceBar";
import { VerdictBadge } from "@/components/ui/VerdictBadge";

const DEMO_SIMS = "0.94, 0.88, 0.81, 0.77, 0.71";

export function MatchConsole() {
  const router = useRouter();
  const [queryId, setQueryId] = useState("c012_q0001");
  const [sims, setSims] = useState(DEMO_SIMS);
  const [visibility, setVisibility] = useState(0.85);
  const [frames, setFrames] = useState(12);
  const [result, setResult] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const similarities = sims
        .split(",")
        .map((s) => Number.parseFloat(s.trim()))
        .filter((n) => Number.isFinite(n));
      if (similarities.length === 0) throw new Error("enter at least one similarity score");
      setResult(
        await postMatch({
          query_id: queryId,
          similarities,
          candidate_ids: similarities.map((_, i) => `g${String(i).padStart(4, "0")}`),
          visibility,
          n_frames: frames,
        }),
      );
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        router.push("/login");
        return;
      }
      setError((e as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-5 md:grid-cols-[1.05fr_1fr]">
      <section className="card p-5">
        <h2 className="text-sm font-medium">Query</h2>
        <p className="mt-1 text-xs leading-relaxed text-paper/45">
          Visibility comes from the segmentation mask. Lowering it should widen the predicted
          temperature and pull confidence down, even when the similarity scores do not move.
        </p>

        <div className="mt-5 space-y-4">
          <div>
            <label className="label" htmlFor="qid">Query id</label>
            <input
              id="qid"
              className="field mt-1.5 font-mono"
              value={queryId}
              maxLength={128}
              onChange={(e) => setQueryId(e.target.value)}
            />
          </div>

          <div>
            <label className="label" htmlFor="sims">Top-k similarities (descending)</label>
            <input
              id="sims"
              className="field mt-1.5 font-mono"
              value={sims}
              onChange={(e) => setSims(e.target.value)}
            />
          </div>

          <div>
            <div className="flex items-baseline justify-between">
              <label className="label" htmlFor="vis">Visible fraction</label>
              <span className="font-mono text-xs text-paper/60">{visibility.toFixed(2)}</span>
            </div>
            <input
              id="vis"
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={visibility}
              onChange={(e) => setVisibility(Number(e.target.value))}
              className="mt-2 w-full accent-accent"
            />
          </div>

          <div>
            <label className="label" htmlFor="frames">Tracklet length (frames)</label>
            <input
              id="frames"
              type="number"
              min={1}
              max={10000}
              className="field mt-1.5 font-mono"
              value={frames}
              onChange={(e) => setFrames(Number(e.target.value))}
            />
          </div>

          <button
            onClick={submit}
            disabled={busy}
            className="w-full btn-primary font-bold tracking-widest disabled:opacity-50"
          >
            {busy ? "SCORING MATCH..." : "SCORE THIS MATCH"}
          </button>
        </div>
      </section>

      <section className="card flex flex-col p-5">
        <h2 className="text-sm font-medium">Decision</h2>

        {error && (
          <p className="mt-4 rounded-lg border border-verdict-reject/30 bg-verdict-reject/10 p-3 text-xs text-verdict-reject">
            {error}
          </p>
        )}

        {!result && !error && (
          <p className="mt-4 text-xs text-paper/40">
            Submit a query to see the calibrated verdict.
          </p>
        )}

        {result && (
          <div className="mt-4 space-y-5">
            <VerdictBadge verdict={result.verdict} />
            <ConfidenceBar confidence={result.confidence} />

            <dl className="grid grid-cols-2 gap-3 text-xs">
              <Stat label="Matched id" value={result.matched_id ?? "—"} />
              <Stat label="Temperature" value={result.temperature.toFixed(3)} />
              <Stat label="Model" value={result.model_version} />
              <Stat label="Raw top-1" value={sims.split(",")[0]?.trim() ?? "—"} />
            </dl>

            <div className="rounded-lg border border-ink-line bg-ink/60 p-3">
              <p className="label">Why</p>
              <p className="mt-1.5 text-xs leading-relaxed text-paper/70">{result.reason}</p>
            </div>

            {result.verdict === "UNCERTAIN" && (
              <div className="flex gap-2">
                <button className="flex-1 rounded-lg border border-verdict-match/40 px-3 py-2 text-xs text-verdict-match transition hover:bg-verdict-match/10">
                  Confirm match
                </button>
                <button className="flex-1 rounded-lg border border-verdict-reject/40 px-3 py-2 text-xs text-verdict-reject transition hover:bg-verdict-reject/10">
                  Reject
                </button>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-1 truncate font-mono text-paper/85">{value}</dd>
    </div>
  );
}

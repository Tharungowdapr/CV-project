import type { Verdict } from "@/types";

const STYLES: Record<Verdict, { ring: string; dot: string; label: string; shadow: string }> = {
  MATCH: { ring: "border-verdict-match text-verdict-match", dot: "bg-verdict-match shadow-[0_0_8px_#00ff9d]", label: "MATCH DETECTED", shadow: "shadow-[0_0_15px_rgba(0,255,157,0.2)]" },
  UNCERTAIN: {
    ring: "border-verdict-uncertain text-verdict-uncertain",
    dot: "bg-verdict-uncertain shadow-[0_0_8px_#ffb800]",
    label: "HUMAN REVIEW REQUIRED",
    shadow: "shadow-[0_0_15px_rgba(255,184,0,0.2)]"
  },
  REJECT: { ring: "border-verdict-reject text-verdict-reject", dot: "bg-verdict-reject shadow-[0_0_8px_#ff3366]", label: "REJECTED / NO MATCH", shadow: "shadow-[0_0_15px_rgba(255,51,102,0.2)]" },
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const s = STYLES[verdict];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-bold tracking-widest bg-ink/50 backdrop-blur-md ${s.ring} ${s.shadow}`}
    >
      <span className={`h-2 w-2 rounded-full ${s.dot}`} aria-hidden />
      {s.label}
    </span>
  );
}

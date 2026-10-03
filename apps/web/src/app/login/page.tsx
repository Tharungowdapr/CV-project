"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { login } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(apiKey);
      router.push("/");
      router.refresh();
    } catch {
      // Deliberately generic: the API itself never distinguishes "wrong key"
      // from "unknown key", so the UI should not invent a distinction either.
      setError("Invalid key");
    } finally {
      setBusy(false);
      setApiKey("");
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-6">
      <div className="card p-6">
        <h1 className="text-lg font-semibold">Sign in</h1>
        <p className="mt-1.5 text-xs leading-relaxed text-paper/50">
          Your key is exchanged once for a session cookie and never stored in the browser.
        </p>

        <form onSubmit={submit} className="mt-5 space-y-4">
          <div>
            <label className="label" htmlFor="key">API key</label>
            <input
              id="key"
              type="password"
              autoComplete="off"
              autoFocus
              className="field mt-1.5 font-mono"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>

          {error && (
            <p className="rounded-lg border border-verdict-reject/30 bg-verdict-reject/10 p-2.5 text-xs text-verdict-reject">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy || !apiKey}
            className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-ink
                       transition hover:bg-accent/90 disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </main>
  );
}

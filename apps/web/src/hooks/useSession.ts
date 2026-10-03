"use client";

import { useCallback, useState } from "react";
import { login as apiLogin, logout as apiLogout } from "@/lib/api";

/**
 * Tracks only whether a session exists client-side, never the credential
 * itself - the source of truth is the httpOnly cookie the server manages.
 * On page reload this resets to "unknown" until the first authenticated
 * request succeeds or fails; there is deliberately no "remember me across
 * reloads" client flag to keep in sync with server-side expiry.
 */
export function useSession() {
  const [role, setRole] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const login = useCallback(async (apiKey: string) => {
    setBusy(true);
    setError(null);
    try {
      const res = await apiLogin(apiKey);
      setRole(res.role);
      return true;
    } catch (e) {
      setError((e as Error).message);
      setRole(null);
      return false;
    } finally {
      setBusy(false);
    }
  }, []);

  const logout = useCallback(async () => {
    await apiLogout().catch(() => undefined); // best-effort; clear local state regardless
    setRole(null);
  }, []);

  const onUnauthorized = useCallback(() => setRole(null), []);

  return { role, error, busy, login, logout, onUnauthorized };
}

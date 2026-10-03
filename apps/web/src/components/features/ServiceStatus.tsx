"use client";

import { useRouter } from "next/navigation";
import { useHealth } from "@/hooks/useHealth";
import { logout } from "@/lib/api";

export function ServiceStatus() {
  const { health, error } = useHealth();
  const router = useRouter();
  const ok = Boolean(health && !error);

  async function handleLogout() {
    await logout().catch(() => undefined);
    router.push("/login");
  }

  return (
    <div className="card flex items-center gap-3 px-4 py-2.5">
      <span
        className={`h-2 w-2 rounded-full ${ok ? "bg-verdict-match" : "bg-verdict-reject"}`}
        aria-hidden
      />
      <div className="text-xs leading-tight">
        <p className="text-paper/80">{ok ? "Service online" : "Service unreachable"}</p>
        <p className="font-mono text-[10px] text-paper/40">
          {health ? `${health.calibrator} · v${health.version}` : (error ?? "checking…")}
        </p>
      </div>
      {/* Sign out removed */}
    </div>
  );
}

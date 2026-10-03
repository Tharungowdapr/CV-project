"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import type { HealthResponse } from "@/types";

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    
    const checkHealth = () => {
      getHealth()
        .then((h) => {
          if (!cancelled) {
            setHealth(h);
            setError(null);
          }
        })
        .catch((e: Error) => {
          if (!cancelled) {
            setError(e.message);
            setHealth(null);
          }
        });
    };

    checkHealth(); // initial check
    const interval = setInterval(checkHealth, 5000); // poll every 5s

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { health, error };
}

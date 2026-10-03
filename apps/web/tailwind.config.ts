import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#050810", soft: "#0a0f1d", line: "#1a253c" },
        paper: "#e2e8f0",
        accent: { DEFAULT: "#00f0ff", glow: "rgba(0, 240, 255, 0.5)", dark: "#008899" },
        purple: { DEFAULT: "#b026ff", glow: "rgba(176, 38, 255, 0.5)" },
        verdict: { match: "#00ff9d", uncertain: "#ffb800", reject: "#ff3366" },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Inter", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;

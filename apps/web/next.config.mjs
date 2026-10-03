/** @type {import('next').NextConfig} */
const isDev = process.env.NODE_ENV !== "production";

export default {
  reactStrictMode: true,
  poweredByHeader: false,
  output: "standalone",
  // No CSP headers in dev — they block cross-origin API calls in subtle ways.
  // In production, add them back via a reverse proxy or middleware.
  async headers() {
    if (isDev) return [];
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "no-referrer" },
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob:",
              `connect-src 'self' ${apiUrl}`,
              "frame-ancestors 'none'",
              "base-uri 'self'",
            ].join("; "),
          },
        ],
      },
    ];
  },
};

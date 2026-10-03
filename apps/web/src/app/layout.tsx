import type { Metadata } from "next";
import Link from "next/link";
import { ServiceStatus } from "@/components/features/ServiceStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vehicle Re-ID — Surveillance System",
  description: "Jarvis-style vehicle tracking and re-identification system.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <header className="sticky top-0 z-50 w-full border-b border-accent/20 bg-ink-soft/40 backdrop-blur-md shadow-[0_0_15px_rgba(0,240,255,0.05)]">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 h-16">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-full bg-accent/20 border border-accent flex items-center justify-center shadow-[0_0_10px_rgba(0,240,255,0.5)]">
                <div className="w-3 h-3 rounded-full bg-accent animate-pulse"></div>
              </div>
              <span className="text-xl font-bold tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-accent to-purple">
                RE-ID SYSTEM
              </span>
            </Link>
            <nav className="flex space-x-6 text-sm uppercase tracking-widest font-semibold text-accent/70 items-center">
              <Link href="/" className="hover:text-accent transition-colors hover:drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">Dashboard</Link>
              <Link href="/dataset" className="hover:text-accent transition-colors hover:drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">Dataset Manager</Link>
              <Link href="/ingest" className="hover:text-accent transition-colors hover:drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">Upload Video</Link>
              <Link href="/live" className="hover:text-accent transition-colors hover:drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">Live Stream</Link>
              <Link href="/search" className="hover:text-accent transition-colors hover:drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">Search Console</Link>
              <div className="pl-4 border-l border-accent/30">
                <ServiceStatus />
              </div>
            </nav>
          </div>
        </header>
        <main className="flex-1 w-full mx-auto max-w-7xl px-6 py-10 relative">
          {children}
        </main>
      </body>
    </html>
  );
}

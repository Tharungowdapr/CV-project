import { Activity, Camera, ShieldAlert, Cpu } from "lucide-react";
import Link from "next/link";

export default function Page() {
  return (
    <main className="space-y-8 animate-in fade-in duration-1000">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-accent/20 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-paper to-accent">SYSTEM DASHBOARD</h1>
          <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-accent/60">
            Real-time RF-DETR object detection and TransReID vector matching console.
          </p>
        </div>
        <div className="flex items-center space-x-2 bg-ink px-4 py-2 rounded-lg border border-accent/30 shadow-[0_0_10px_rgba(0,240,255,0.1)]">
          <div className="w-2 h-2 rounded-full bg-verdict-match animate-pulse"></div>
          <span className="text-xs uppercase tracking-widest text-verdict-match font-bold">System Online</span>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card p-6 flex flex-col items-center justify-center text-center space-y-4 hover:border-accent/50 transition-colors">
          <Camera className="w-8 h-8 text-purple" />
          <div>
            <div className="text-3xl font-light text-paper">12</div>
            <div className="label mt-1">Active Streams</div>
          </div>
        </div>
        
        <div className="card p-6 flex flex-col items-center justify-center text-center space-y-4 hover:border-accent/50 transition-colors">
          <Cpu className="w-8 h-8 text-accent" />
          <div>
            <div className="text-3xl font-light text-paper">4.2ms</div>
            <div className="label mt-1">Avg Inference</div>
          </div>
        </div>

        <div className="card p-6 flex flex-col items-center justify-center text-center space-y-4 hover:border-accent/50 transition-colors">
          <ShieldAlert className="w-8 h-8 text-verdict-uncertain" />
          <div>
            <div className="text-3xl font-light text-paper">8</div>
            <div className="label mt-1">Pending Reviews</div>
          </div>
        </div>

        <div className="card p-6 flex flex-col items-center justify-center text-center space-y-4 hover:border-accent/50 transition-colors">
          <Activity className="w-8 h-8 text-verdict-match" />
          <div>
            <div className="text-3xl font-light text-paper">99.8%</div>
            <div className="label mt-1">Detection Rate</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
        <div className="card p-8 border-accent/40 bg-ink/80">
          <h2 className="text-xl font-bold tracking-widest text-accent mb-4">DATASET MANAGER</h2>
          <p className="text-sm text-paper/70 mb-6">Upload videos, RTSP streams, or image directories for offline tracking and Re-ID indexing using SAM 2.1 and RF-DETR.</p>
          <Link href="/dataset" className="btn-primary">Manage Datasets</Link>
        </div>

        <div className="card p-8 border-purple/40 bg-ink/80">
          <h2 className="text-xl font-bold tracking-widest text-purple mb-4">SEARCH CONSOLE</h2>
          <p className="text-sm text-paper/70 mb-6">Perform vector similarity searches across the pgvector database using attributes, plates, and photos.</p>
          <Link href="/search" className="btn-primary border-purple text-purple hover:bg-purple/20 hover:shadow-[0_0_15px_rgba(176,38,255,0.4)]">Launch Search</Link>
        </div>
      </div>

      <footer className="border-t border-accent/20 pt-5 text-xs tracking-widest text-accent/40 flex justify-between">
        <span>PROJECT: VEHICLE RE-ID UNCERTAINTY</span>
        <span>AUTH: ROOT_ACCESS</span>
      </footer>
    </main>
  );
}

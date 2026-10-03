"use client";

import { useEffect, useState, useRef } from "react";
import { configureLiveStream } from "@/lib/api";

export default function LivePage() {
  const [isConnected, setIsConnected] = useState(false);
  const [vehicles, setVehicles] = useState<any[]>([]);
  const [fps, setFps] = useState(0);
  const [streamUrl, setStreamUrl] = useState("0");
  const [isConfiguring, setIsConfiguring] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const frameCount = useRef(0);
  const lastTime = useRef(Date.now());

  useEffect(() => {
    // Connect to WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_API_URL 
      ? process.env.NEXT_PUBLIC_API_URL.replace("http", "ws") 
      : "ws://localhost:8000";
      
    const ws = new WebSocket(`${wsUrl}/api/live/ws/vehicles`);

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setVehicles(data.vehicles || []);
      
      // Calculate FPS
      frameCount.current++;
      const now = Date.now();
      if (now - lastTime.current >= 1000) {
        setFps(frameCount.current);
        frameCount.current = 0;
        lastTime.current = now;
      }
      
      // Draw image
      if (data.image && canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        const img = new Image();
        img.onload = () => {
          // Adjust canvas size
          canvasRef.current!.width = img.width;
          canvasRef.current!.height = img.height;
          ctx?.drawImage(img, 0, 0);
        };
        img.src = `data:image/jpeg;base64,${data.image}`;
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  async function applyStreamUrl(url: string) {
    setIsConfiguring(true);
    try {
      setStreamUrl(url);
      await configureLiveStream(url);
    } catch (e) {
      alert("Failed to configure stream: " + (e as Error).message);
    } finally {
      setIsConfiguring(false);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-7xl space-y-8">
          <header className="flex flex-col gap-2 flex-row justify-between items-end">
            <div>
              <h1 className="text-3xl font-light tracking-tight text-paper">
                Live Traffic Stream
              </h1>
              <p className="text-sm text-paper/60">
                Real-time tracking and Re-ID processing.
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className={`h-3 w-3 rounded-full ${isConnected ? 'bg-verdict-match animate-pulse' : 'bg-verdict-reject'}`}></span>
                <span className="text-sm uppercase tracking-widest text-accent/70">{isConnected ? 'Live' : 'Disconnected'}</span>
              </div>
              <div className="text-xs font-mono text-paper/50">FPS: {fps}</div>
            </div>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            <div className="md:col-span-12 lg:col-span-8">
              <div className="card overflow-hidden border-accent/20 bg-ink-soft/40 relative h-full">
                {!isConnected && (
                  <div className="absolute inset-0 flex items-center justify-center bg-ink/80 backdrop-blur-sm z-10">
                    <div className="text-center p-8 rounded-2xl border border-accent/10 bg-ink/50 shadow-2xl">
                      <div className="animate-spin rounded-full h-10 w-10 border-2 border-accent border-t-transparent mx-auto mb-4"></div>
                      <p className="text-accent tracking-widest text-sm font-medium">CONNECTING TO STREAM...</p>
                      <p className="text-xs text-paper/40 mt-2 font-mono">AWAITING WEBSOCKET HANDSHAKE</p>
                    </div>
                  </div>
                )}
                <canvas 
                  ref={canvasRef} 
                  className="w-full aspect-video object-contain bg-black shadow-inner"
                />
              </div>
            </div>

            <div className="md:col-span-12 lg:col-span-4 space-y-6 flex flex-col">
              {/* Control Panel */}
              <div className="card p-5 border-accent/30 bg-gradient-to-b from-ink-soft/80 to-ink shadow-lg">
                <h3 className="text-xs font-bold uppercase tracking-widest text-accent mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
                  Stream Configuration
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-[10px] text-paper/50 uppercase tracking-wider mb-1.5 block">Source URL (RTSP / HTTP / 0)</label>
                    <div className="flex gap-2">
                      <input 
                        type="text" 
                        value={streamUrl} 
                        onChange={e => setStreamUrl(e.target.value)} 
                        className="field font-mono text-xs w-full bg-ink border-accent/20 focus:border-accent text-paper py-2" 
                        placeholder="e.g. 0 or rtsp://..."
                      />
                      <button 
                        onClick={() => applyStreamUrl(streamUrl)} 
                        disabled={isConfiguring}
                        className="btn-primary py-2 px-4 text-xs font-bold disabled:opacity-50 whitespace-nowrap shadow-accent/20 shadow-lg"
                      >
                        {isConfiguring ? "..." : "Apply"}
                      </button>
                    </div>
                  </div>
                  
                  <div className="pt-2 border-t border-ink-line">
                    <label className="text-[10px] text-paper/50 uppercase tracking-wider mb-2 block">Quick Connect</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button 
                        onClick={() => applyStreamUrl("0")} 
                        className="py-2 px-3 text-xs bg-ink/50 border border-ink-line rounded hover:border-accent/50 hover:bg-accent/5 text-paper/80 hover:text-accent transition-all flex justify-center items-center gap-2"
                      >
                        Local Webcam
                      </button>
                      <button 
                        onClick={() => applyStreamUrl("https://storage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4")} 
                        className="py-2 px-3 text-xs bg-ink/50 border border-ink-line rounded hover:border-accent/50 hover:bg-accent/5 text-paper/80 hover:text-accent transition-all flex justify-center items-center gap-2"
                      >
                        Test Stream
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Stats Panel */}
              <div className="card p-5 border-accent/20 flex-1 flex flex-col">
                <h3 className="text-xs font-bold uppercase tracking-widest text-paper/70 mb-4">Live Tracking Stats</h3>
                <div className="flex justify-between items-end pb-4 border-b border-ink-line mb-4">
                  <span className="text-sm text-paper/60">Active Subjects</span>
                  <span className="text-3xl font-light text-accent">{vehicles.length}</span>
                </div>
                
                <div className="space-y-2 overflow-y-auto pr-2 flex-1 max-h-[300px]">
                  {vehicles.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center opacity-50 py-8">
                      <div className="w-8 h-8 border border-dashed border-paper/30 rounded-full mb-3 flex items-center justify-center">
                        <span className="text-xs text-paper/50">0</span>
                      </div>
                      <span className="text-xs text-paper/40 font-mono">NO VEHICLES IN FRAME</span>
                    </div>
                  ) : (
                    vehicles.map((v, i) => (
                      <div key={i} className="p-3 rounded-lg border border-accent/10 bg-ink-soft/30 flex justify-between items-center hover:border-accent/30 hover:bg-accent/5 transition-colors">
                        <div>
                          <div className="text-xs font-mono text-paper font-semibold">{v.vehicle_id}</div>
                          <div className="text-[9px] uppercase tracking-wider text-paper/40 mt-1">{v.type || 'Vehicle'} • Trk #{v.track_id}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[9px] text-accent/50 uppercase tracking-widest">Conf</div>
                          <div className="text-xs font-mono text-accent font-bold">{(v.confidence * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

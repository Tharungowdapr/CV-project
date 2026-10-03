"use client";

import { useState, useRef, useEffect } from "react";
import { ingestVideo, getIngestionStatus, IngestionStatus } from "@/lib/api";

export default function IngestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<IngestionStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Poll status when jobId is set
  useEffect(() => {
    if (!jobId || (status?.status === 'done' || status?.status === 'error')) return;
    
    const interval = setInterval(async () => {
      try {
        const s = await getIngestionStatus(jobId);
        setStatus(s);
      } catch (err: any) {
        console.error("Failed to poll status:", err);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [jobId, status?.status]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0] ?? null);
      setError(null);
      setJobId(null);
      setStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const res = await ingestVideo(file);
      setJobId(res.job_id);
      setStatus(res);
    } catch (err: any) {
      setError(err.message || "Failed to start ingestion");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-3xl space-y-8">
          <header className="flex flex-col gap-2">
            <h1 className="text-3xl font-light tracking-tight text-paper">
              Video Ingestion
            </h1>
            <p className="text-sm text-paper/60">
              Upload CCTV footage to automatically extract and index vehicle sightings using the 2-stage (RF-DETR + OSNet) pipeline.
            </p>
          </header>

          <div className="card p-6">
            <div className="space-y-6">
              {/* Upload Zone */}
              <div 
                className={`relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-colors ${
                  file ? 'border-accent/50 bg-accent/5' : 'border-ink-line bg-ink hover:border-accent/30 hover:bg-ink-soft'
                }`}
                onClick={() => !uploading && fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef}
                  className="hidden" 
                  accept="video/*"
                  onChange={handleFileChange}
                  disabled={uploading || status?.status === 'running'}
                />
                
                {file ? (
                  <div className="text-center">
                    <div className="mb-2 text-4xl text-accent">📄</div>
                    <div className="font-medium text-paper">{file.name}</div>
                    <div className="text-xs text-paper/50">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="mb-2 text-4xl text-paper/20">📹</div>
                    <div className="font-medium text-paper">Click to upload video</div>
                    <div className="text-xs text-paper/50">MP4, AVI, MKV up to 500MB</div>
                  </div>
                )}
              </div>

              {error && (
                <div className="rounded-lg bg-verdict-reject/10 p-3 text-sm text-verdict-reject border border-verdict-reject/30">
                  {error}
                </div>
              )}

              {/* Upload Button */}
              {!jobId && (
                <button
                  className="btn-primary w-full py-3"
                  onClick={handleUpload}
                  disabled={!file || uploading}
                >
                  {uploading ? (
                    <span className="flex items-center gap-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Uploading...
                    </span>
                  ) : "Start Processing"}
                </button>
              )}

              {/* Progress Tracking */}
              {status && (
                <div className="space-y-4 rounded-lg bg-ink-soft/30 p-5 border border-ink-line">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-paper uppercase tracking-wider">
                      Status: <span className={
                        status.status === 'running' ? 'text-accent' : 
                        status.status === 'done' ? 'text-verdict-match' : 
                        status.status === 'error' ? 'text-verdict-reject' : 'text-paper'
                      }>{status.status}</span>
                    </h3>
                    {status.eta_seconds !== null && status.status === 'running' && (
                      <span className="text-xs font-mono text-accent">
                        ETA: {Math.ceil(status.eta_seconds ?? 0)}s
                      </span>
                    )}
                  </div>
                  
                  {status.status === 'running' || status.status === 'done' ? (
                    <>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-ink">
                        <div 
                          className="h-full bg-accent transition-all duration-300"
                          style={{ 
                            width: `${status.total_frames ? Math.min(100, Math.round((status.processed_frames! / status.total_frames) * 100)) : 0}%` 
                          }}
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 pt-2 sm:grid-cols-4">
                        <div>
                          <div className="text-[10px] text-paper/50 uppercase">Frames</div>
                          <div className="font-mono text-sm text-paper">{status.processed_frames} / {status.total_frames}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-paper/50 uppercase">Vehicles Detected</div>
                          <div className="font-mono text-sm text-paper">{status.detections}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-paper/50 uppercase">Vectors Indexed</div>
                          <div className="font-mono text-sm text-paper">{status.embeddings}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-paper/50 uppercase">Elapsed</div>
                          <div className="font-mono text-sm text-paper">{Math.round(status.elapsed_seconds || 0)}s</div>
                        </div>
                      </div>
                    </>
                  ) : null}

                  {status.error && (
                    <div className="mt-2 text-sm text-verdict-reject">
                      Error: {status.error}
                    </div>
                  )}

                  {status.status === 'done' && (
                    <div className="mt-6 border-t border-ink-line pt-4 space-y-4">
                      <div className="flex items-center gap-2 text-sm font-medium text-verdict-match">
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Processing complete. Vectors added to database.
                      </div>
                      <div className="card p-4 border-accent/20 bg-ink">
                        <h4 className="text-sm font-medium text-accent mb-3">Video Summary</h4>
                        <div className="space-y-2 text-xs text-paper/80">
                          <div className="flex justify-between"><span>File:</span> <span className="font-mono text-paper">{status.filename}</span></div>
                          <div className="flex justify-between"><span>Total Frames Processed:</span> <span className="font-mono text-paper">{status.processed_frames}</span></div>
                          <div className="flex justify-between"><span>Unique Vehicle Sightings Indexed:</span> <span className="font-mono text-paper">{status.embeddings}</span></div>
                          <div className="flex justify-between"><span>Total Processing Time:</span> <span className="font-mono text-paper">{Math.round(status.elapsed_seconds || 0)} seconds</span></div>
                          <div className="flex justify-between"><span>Processing Speed:</span> <span className="font-mono text-paper">{((status.processed_frames || 1) / Math.max(1, status.elapsed_seconds || 1)).toFixed(1)} fps</span></div>
                        </div>
                      </div>
                      <div className="pt-2 flex gap-4">
                        <button onClick={() => window.location.href='/database'} className="btn-primary flex-1 py-2 text-xs">View in Database</button>
                        <button onClick={() => { setFile(null); setJobId(null); setStatus(null); }} className="btn-primary flex-1 py-2 text-xs bg-ink border border-accent/30 text-accent hover:bg-accent/10">Upload Another Video</button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

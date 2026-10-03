"use client";

import { useEffect, useState } from "react";
import { getGalleryEntries, getGalleryStats, deleteDataset, GalleryEntry } from "@/lib/api";
import { Trash2 } from "lucide-react";

export default function DatabasePage() {
  const [stats, setStats] = useState<{ total: number; cameras: number; datasets: number; plates: number } | null>(null);
  const [entries, setEntries] = useState<GalleryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    try {
      const [statsData, entriesData] = await Promise.all([
        getGalleryStats(),
        getGalleryEntries(500, 0)
      ]);
      setStats(statsData);
      setEntries(entriesData);
    } catch (err: any) {
      setError(err.message || "Failed to load database entries");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const datasets = Array.from(new Set(entries.map((e) => e.dataset)));
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets[0] ?? null);
    }
  }, [datasets, selectedDataset]);

  async function handleDeleteDataset(ds: string) {
    if (!confirm(`Are you sure you want to delete dataset '${ds}'?`)) return;
    try {
      await deleteDataset(ds);
      if (selectedDataset === ds) setSelectedDataset(null);
      await loadData();
    } catch (err: any) {
      alert(`Failed to delete: ${err.message}`);
    }
  }

  const filteredEntries = entries.filter((e) => e.dataset === selectedDataset);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-7xl space-y-8">
          <header className="flex flex-col gap-2">
            <h1 className="text-3xl font-light tracking-tight text-paper">
              Database Explorer
            </h1>
            <p className="text-sm text-paper/60">
              Hierarchical view of indexed gallery vectors and metadata across datasets.
            </p>
          </header>

          {error ? (
            <div className="rounded-xl border border-verdict-reject/30 bg-verdict-reject/10 p-4 text-sm text-verdict-reject">
              {error}
            </div>
          ) : loading ? (
            <div className="flex items-center justify-center p-12">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            </div>
          ) : (
            <div className="space-y-6">
              {/* Stats Grid */}
              {stats && (
                <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                  {[
                    { label: "Total Entries", value: stats.total },
                    { label: "Datasets", value: stats.datasets },
                    { label: "Cameras", value: stats.cameras },
                    { label: "Plates Indexed", value: stats.plates },
                  ].map((stat, i) => (
                    <div key={i} className="card p-5">
                      <div className="label text-accent/50">{stat.label}</div>
                      <div className="mt-2 text-3xl font-light text-paper">{stat.value.toLocaleString()}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Dataset Selector */}
              <div className="flex flex-wrap gap-2 border-b border-ink-line pb-4">
                {datasets.map((ds) => (
                  <button
                    key={ds}
                    onClick={() => setSelectedDataset(ds)}
                    className={`rounded-full px-4 py-1.5 text-sm transition-colors ${
                      selectedDataset === ds 
                        ? "bg-accent text-ink font-medium" 
                        : "bg-ink-soft text-paper/60 hover:bg-ink-line hover:text-paper"
                    }`}
                  >
                    {ds}
                  </button>
                ))}
                {datasets.length === 0 && (
                  <div className="text-sm text-paper/50">No datasets available.</div>
                )}
              </div>

              {/* Entries for selected dataset */}
              {selectedDataset && (
                <div className="space-y-6">
                  <div className="card p-5 border-accent/20 bg-ink-soft/40">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h2 className="text-xl font-light text-paper">
                          Dataset: <span className="text-accent">{selectedDataset}</span>
                        </h2>
                        <p className="text-sm text-paper/60">Executive Summary & Export</p>
                      </div>
                      <div className="flex gap-2">
                        <button 
                          onClick={() => {
                            // Simple CSV export
                            const headers = ["UID", "Dataset", "Camera", "Plate", "ID", "Confidence"];
                            const csvContent = "data:text/csv;charset=utf-8," 
                              + headers.join(",") + "\n"
                              + filteredEntries.map(e => `${e.uid},${e.dataset},${e.camera_id},${e.plate_text||""},${e.ground_truth_identity||""},${e.visibility||""}`).join("\n");
                            const encodedUri = encodeURI(csvContent);
                            const link = document.createElement("a");
                            link.setAttribute("href", encodedUri);
                            link.setAttribute("download", `${selectedDataset}_export.csv`);
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                          }}
                          className="btn-primary py-1.5 px-4 text-xs flex items-center gap-2"
                        >
                          Export to CSV
                        </button>
                        <button 
                          onClick={() => handleDeleteDataset(selectedDataset)}
                          className="flex items-center gap-1.5 rounded bg-verdict-reject/10 px-3 py-1.5 text-xs text-verdict-reject hover:bg-verdict-reject/20 transition border border-verdict-reject/30"
                        >
                          <Trash2 className="h-3 w-3" />
                          Delete
                        </button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-ink-line">
                      <div>
                        <div className="text-[10px] text-paper/50 uppercase">Total Entries</div>
                        <div className="font-mono text-xl text-paper">{filteredEntries.length}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-paper/50 uppercase">Unique Cameras</div>
                        <div className="font-mono text-xl text-paper">{new Set(filteredEntries.map(e => e.camera_id)).size}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-paper/50 uppercase">Plates Detected</div>
                        <div className="font-mono text-xl text-paper">{filteredEntries.filter(e => e.plate_text).length}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-paper/50 uppercase">Identities (IDs)</div>
                        <div className="font-mono text-xl text-paper">{new Set(filteredEntries.map(e => e.ground_truth_identity).filter(Boolean)).size}</div>
                      </div>
                    </div>
                  </div>

                  <div className="card overflow-hidden">
                    <div className="border-b border-ink-line bg-ink-soft/50 px-5 py-3 flex justify-between items-center">
                      <h3 className="text-sm font-medium text-paper">Dataset Vectors ({filteredEntries.length})</h3>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4">
                        {filteredEntries.map((entry) => (
                          <div key={entry.uid} className="group relative overflow-hidden rounded-xl bg-ink border border-ink-line transition hover:border-accent/50">
                            <div className="aspect-[4/3] w-full bg-ink-soft relative">
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={`http://localhost:8000/api/files/${encodeURIComponent(entry.image_path)}`}
                                alt={entry.uid}
                                className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                                onError={(e) => {
                                  (e.target as HTMLImageElement).style.display = 'none';
                                  (e.target as HTMLImageElement).parentElement!.classList.add('flex', 'items-center', 'justify-center');
                                  (e.target as HTMLImageElement).parentElement!.innerHTML = '<span class="text-xs text-paper/30">Missing Image</span>';
                                }}
                              />
                              <div className="absolute top-2 right-2 flex flex-col gap-1 items-end">
                                {entry.ground_truth_identity && (
                                   <div className="bg-ink/90 backdrop-blur text-[10px] text-accent px-2 py-0.5 rounded border border-accent/30 font-mono shadow">
                                     VID: {entry.ground_truth_identity}
                                   </div>
                                )}
                                <div className="bg-ink/90 backdrop-blur text-[10px] text-purple px-2 py-0.5 rounded border border-purple/30 font-mono shadow">
                                  PROB: {((entry.visibility ?? 1.0) * 100).toFixed(0)}%
                                </div>
                              </div>
                            </div>
                            <div className="p-4 space-y-3">
                              <div className="flex items-center justify-between border-b border-ink-line pb-2">
                                <span className="font-mono text-xs text-paper font-medium">{entry.uid.split('_').pop()}</span>
                                {entry.plate_text && (
                                  <span className="rounded bg-accent/10 px-2 py-0.5 font-mono text-[10px] text-accent border border-accent/20">
                                    {entry.plate_text}
                                  </span>
                                )}
                              </div>
                              
                              <div className="space-y-1.5">
                                <div className="flex justify-between text-[11px]">
                                  <span className="text-paper/40">Camera:</span>
                                  <span className="text-paper/80 font-mono">{entry.camera_id}</span>
                                </div>
                                <div className="flex justify-between text-[11px]">
                                  <span className="text-paper/40">Indexed at:</span>
                                  <span className="text-paper/80">{entry.indexed_at ? new Date(entry.indexed_at).toLocaleDateString() : 'N/A'}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

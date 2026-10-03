"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  listDatasets,
  ingestDatasetZip,
  getDatasetIngestStatus,
  getDatasetRows,
  downloadDatasetCsv,
  checkBackendHealth,
  DatasetListEntry,
  DatasetIngestStatus,
  BackendHealth,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Utilities ─────────────────────────────────────────────────────────────

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

function fmtDate(iso: string): string {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

// Column display names
const COL_LABELS: Record<string, string> = {
  uid: "UID", dataset_name: "Dataset", split: "Split", relative_path: "Path",
  identity: "Identity", camera_id: "Camera", frame_id: "Frame", sequence_id: "Seq",
  vehicle_type: "Type", color: "Color", brand: "Brand", plate_text: "Plate",
  plate_confidence: "Plate Conf", visibility: "Visibility", width: "W", height: "H",
  file_size_bytes: "Size (B)", md5: "MD5", is_preview: "Preview",
  preview_path: "Preview Path", timestamp: "Timestamp",
};

// Column widths (px)
const COL_WIDTHS: Record<string, number> = {
  uid: 220, dataset_name: 100, split: 70, relative_path: 200,
  identity: 70, camera_id: 80, frame_id: 80, sequence_id: 50,
  vehicle_type: 90, color: 75, brand: 85, plate_text: 130,
  plate_confidence: 100, visibility: 80, width: 55, height: 55,
  file_size_bytes: 80, md5: 130, is_preview: 75, preview_path: 180, timestamp: 160,
};

// Columns that should show a colored badge
const BADGE_COLS = new Set(["split", "vehicle_type", "color", "is_preview"]);

// Badge color map
const SPLIT_COLORS: Record<string, string> = {
  train: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  query: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  gallery: "bg-green-500/20 text-green-300 border-green-500/30",
  all: "bg-accent/20 text-accent border-accent/30",
};

function CellBadge({ col, value }: { col: string; value: string }) {
  if (col === "split") {
    const cls = SPLIT_COLORS[value] ?? "bg-ink-line text-paper/60 border-ink-line";
    return <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium border ${cls}`}>{value || "—"}</span>;
  }
  if (col === "is_preview") {
    return value === "True" || value === "true"
      ? <span className="text-accent text-[10px] font-bold">✓ YES</span>
      : <span className="text-paper/30 text-[10px]">—</span>;
  }
  if (col === "vehicle_type") {
    const icons: Record<string, string> = { sedan: "🚗", suv: "🚙", truck: "🚚", van: "🚐", bus: "🚌", motorcycle: "🏍", pickup: "🛻" };
    return <span className="text-[11px] text-paper/80">{icons[value] ?? "🚘"} {value}</span>;
  }
  if (col === "color") {
    const swatches: Record<string, string> = {
      black: "#111", white: "#eee", silver: "#aaa", red: "#e44", blue: "#44e",
      green: "#4a4", yellow: "#ee4", gray: "#888", brown: "#964", gold: "#ca4",
    };
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-paper/80">
        <span className="w-3 h-3 rounded-full border border-ink-line" style={{ background: swatches[value] ?? "#666" }} />
        {value}
      </span>
    );
  }
  return <span>{value || "—"}</span>;
}

// ─── Stat Card ──────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }: {
  label: string; value: string | number; sub?: string; accent?: string;
}) {
  return (
    <div className="card p-4 flex flex-col gap-1 min-w-0">
      <span className="text-[10px] tracking-widest text-paper/50 uppercase">{label}</span>
      <span className="text-2xl font-bold font-mono" style={{ color: accent ?? "var(--color-paper)" }}>{value}</span>
      {sub && <span className="text-xs text-paper/40">{sub}</span>}
    </div>
  );
}

// ─── Progress Bar ───────────────────────────────────────────────────────────

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink">
      <div
        className="absolute inset-y-0 left-0 rounded-full bg-accent transition-all duration-300"
        style={{ width: `${pct}%`, boxShadow: "0 0 8px rgba(0,240,255,0.6)" }}
      />
    </div>
  );
}

// ─── Ingest Panel ──────────────────────────────────────────────────────────

function IngestPanel({ onDone }: { onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [split, setSplit] = useState("all");
  const [nPreview, setNPreview] = useState(20);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<DatasetIngestStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPoll = () => { if (pollRef.current) clearInterval(pollRef.current); };

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") {
      stopPoll();
      if (job?.status === "done") onDone();
      return;
    }
    stopPoll();
    pollRef.current = setInterval(async () => {
      try { const s = await getDatasetIngestStatus(job.job_id); setJob(s); } catch {}
    }, 1000);
    return stopPoll;
  }, [job?.job_id, job?.status]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); if (!name) setName(f.name.replace(/\.(zip|tar\.gz)$/i, "").replace(/[^A-Za-z0-9_-]/g, "_")); }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setUploading(true); setError(null);
    try { const result = await ingestDatasetZip(file, name || "dataset", split, nPreview); setJob(result); }
    catch (err: any) { setError(err.message ?? "Upload failed"); }
    finally { setUploading(false); }
  };

  const reset = () => { setFile(null); setName(""); setJob(null); setError(null); };

  return (
    <div className="card p-6 space-y-5">
      <h2 className="text-sm font-semibold tracking-widest text-paper/70 uppercase flex items-center gap-2">
        <span className="text-accent">⬆</span> Ingest New Dataset
      </h2>
      {!job && (
        <>
          <div
            className={`relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-all ${
              dragging ? "border-accent bg-accent/10 scale-[1.01]"
              : file ? "border-accent/50 bg-accent/5"
              : "border-ink-line bg-ink hover:border-accent/30 hover:bg-ink-soft"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !uploading && fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" className="hidden" accept=".zip"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) { setFile(f); if (!name) setName(f.name.replace(/\.zip$/i, "").replace(/[^A-Za-z0-9_-]/g, "_")); }
              }} />
            {file ? (
              <div className="text-center">
                <div className="mb-2 text-5xl">📦</div>
                <div className="font-semibold text-paper">{file.name}</div>
                <div className="text-xs text-paper/50 mt-1">{fmtBytes(file.size)}</div>
              </div>
            ) : (
              <div className="text-center">
                <div className="mb-3 text-5xl text-paper/20">🗂</div>
                <div className="font-medium text-paper">Drop a ZIP archive here</div>
                <div className="text-xs text-paper/50 mt-1">Supports VeRi-776, VeriWild, or any image folder packed as ZIP</div>
              </div>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-[10px] tracking-wider text-paper/50 mb-1 uppercase">Dataset Name</label>
              <input className="w-full rounded-lg border border-ink-line bg-ink px-3 py-2 text-sm text-paper placeholder:text-paper/30 focus:border-accent focus:outline-none"
                placeholder="e.g. veri776_train" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
              <label className="block text-[10px] tracking-wider text-paper/50 mb-1 uppercase">Split Label</label>
              <select className="w-full rounded-lg border border-ink-line bg-ink px-3 py-2 text-sm text-paper focus:border-accent focus:outline-none"
                value={split} onChange={(e) => setSplit(e.target.value)}>
                <option value="all">all</option>
                <option value="train">train</option>
                <option value="query">query</option>
                <option value="gallery">gallery</option>
              </select>
            </div>
            <div>
              <label className="block text-[10px] tracking-wider text-paper/50 mb-1 uppercase">Preview Images</label>
              <input type="number" min={0} max={100}
                className="w-full rounded-lg border border-ink-line bg-ink px-3 py-2 text-sm text-paper focus:border-accent focus:outline-none"
                value={nPreview} onChange={(e) => setNPreview(Number(e.target.value))} />
            </div>
          </div>
          {error && <div className="rounded-lg bg-red-900/20 border border-red-500/30 p-3 text-sm text-red-400">{error}</div>}
          <button className="btn-primary w-full py-3" disabled={!file || uploading} onClick={handleSubmit}>
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Uploading…
              </span>
            ) : "Process Dataset →"}
          </button>
        </>
      )}
      {job && (
        <div className="space-y-4 rounded-lg border border-ink-line bg-ink-soft/30 p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-paper uppercase tracking-wider">
              Status: <span className={job.status === "running" ? "text-accent" : job.status === "done" ? "text-green-400" : job.status === "error" ? "text-red-400" : "text-paper"}>{job.status}</span>
            </h3>
            <span className="text-xs font-mono text-paper/50">{job.elapsed_seconds.toFixed(1)}s</span>
          </div>
          <ProgressBar value={job.processed} max={Math.max(job.total_images, 1)} />
          <div className="grid grid-cols-3 gap-3 text-center">
            {[["Total", job.total_images], ["Processed", job.processed], ["Previews", job.preview_count]].map(([label, val]) => (
              <div key={label as string} className="rounded-lg bg-ink p-3">
                <div className="text-[10px] text-paper/50 uppercase mb-1">{label}</div>
                <div className="font-mono text-sm text-paper">{val}</div>
              </div>
            ))}
          </div>
          {job.status === "done" && (
            <div className="pt-3 border-t border-ink-line space-y-3">
              <div className="flex items-center gap-2 text-sm text-green-400">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Dataset processed — CSV manifest ready
              </div>
              <div className="flex gap-3">
                <a href={downloadDatasetCsv(job.dataset_name)} download className="btn-primary flex-1 py-2 text-xs text-center">⬇ Download CSV</a>
                <button onClick={reset} className="btn-primary flex-1 py-2 text-xs bg-ink border border-accent/30 text-accent hover:bg-accent/10">Process Another</button>
              </div>
            </div>
          )}
          {job.status === "error" && (
            <div className="text-sm text-red-400 bg-red-900/10 rounded-lg p-3 border border-red-500/30">{job.error || "Unknown error"}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Excel-Style Data Table ─────────────────────────────────────────────────

type SortDir = "asc" | "desc" | null;

function ExcelTable({ rows, columns }: { rows: Record<string, string>[]; columns: string[] }) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>(null);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [copiedCell, setCopiedCell] = useState<string | null>(null);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir(d => d === "asc" ? "desc" : d === "desc" ? null : "asc");
      if (sortDir === "desc") setSortCol(null);
    } else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

  const handleFilterChange = (col: string, val: string) => {
    setFilters(f => val ? { ...f, [col]: val } : Object.fromEntries(Object.entries(f).filter(([k]) => k !== col)));
  };

  const filteredRows = useMemo(() => {
    let r = rows;
    for (const [col, val] of Object.entries(filters)) {
      if (!val) continue;
      r = r.filter(row => String(row[col] ?? "").toLowerCase().includes(val.toLowerCase()));
    }
    if (sortCol && sortDir) {
      r = [...r].sort((a, b) => {
        const av = a[sortCol] ?? "", bv = b[sortCol] ?? "";
        const cmp = isNaN(Number(av)) || isNaN(Number(bv))
          ? av.localeCompare(bv) : Number(av) - Number(bv);
        return sortDir === "asc" ? cmp : -cmp;
      });
    }
    return r;
  }, [rows, filters, sortCol, sortDir]);

  const copyCell = (val: string, key: string) => {
    navigator.clipboard.writeText(val).catch(() => {});
    setCopiedCell(key);
    setTimeout(() => setCopiedCell(null), 1200);
  };

  const activeFilters = Object.entries(filters).filter(([, v]) => v);

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <span className="text-xs text-paper/50 font-mono">
            {filteredRows.length.toLocaleString()} of {rows.length.toLocaleString()} rows
          </span>
          {activeFilters.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {activeFilters.map(([col, val]) => (
                <span key={col} className="flex items-center gap-1 bg-accent/10 border border-accent/30 text-accent text-[10px] px-2 py-0.5 rounded-full">
                  {COL_LABELS[col] ?? col}: {val}
                  <button onClick={() => handleFilterChange(col, "")} className="ml-0.5 hover:text-red-400">✕</button>
                </span>
              ))}
              <button onClick={() => setFilters({})} className="text-[10px] text-paper/40 hover:text-red-400 underline ml-1">Clear all</button>
            </div>
          )}
        </div>
        <button
          id="toggle-column-filters"
          onClick={() => setShowFilters(f => !f)}
          className={`px-3 py-1.5 rounded-lg text-xs border transition-all ${
            showFilters ? "border-accent/50 bg-accent/10 text-accent" : "border-ink-line bg-ink text-paper/60 hover:border-accent/30"
          }`}
        >
          {showFilters ? "▲ Hide Filters" : "▼ Column Filters"}
        </button>
      </div>

      {/* Table */}
      <div className="overflow-auto rounded-xl border border-ink-line shadow-lg" style={{ maxHeight: "65vh" }}>
        <table className="text-xs border-collapse" style={{ minWidth: "max-content" }}>
          {/* Sticky Header */}
          <thead className="sticky top-0 z-20">
            {/* Column labels */}
            <tr style={{ background: "linear-gradient(180deg, hsl(220,20%,16%) 0%, hsl(220,20%,12%) 100%)" }}>
              {/* Row number header */}
              <th className="px-2 py-2 text-center text-[10px] text-paper/30 font-medium border-b border-r border-ink-line/50 select-none" style={{ minWidth: 40 }}>#</th>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left font-semibold tracking-wider text-paper/70 uppercase whitespace-nowrap border-b border-r border-ink-line/50 cursor-pointer hover:bg-accent/5 hover:text-accent select-none transition-colors group"
                  style={{ minWidth: COL_WIDTHS[col] ?? 100 }}
                  onClick={() => handleSort(col)}
                >
                  <span className="flex items-center gap-1">
                    {COL_LABELS[col] ?? col}
                    <span className="text-[8px] opacity-50 group-hover:opacity-100 transition-opacity">
                      {sortCol === col ? (sortDir === "asc" ? "▲" : "▼") : "⇅"}
                    </span>
                  </span>
                </th>
              ))}
            </tr>
            {/* Filter row */}
            {showFilters && (
              <tr style={{ background: "hsl(220,20%,10%)" }}>
                <td className="px-2 py-1 border-b border-r border-ink-line/30" />
                {columns.map((col) => (
                  <td key={col} className="px-2 py-1 border-b border-r border-ink-line/30">
                    <input
                      type="text"
                      value={filters[col] ?? ""}
                      onChange={(e) => handleFilterChange(col, e.target.value)}
                      placeholder="Filter…"
                      className="w-full bg-ink-soft/50 border border-ink-line/50 rounded px-2 py-0.5 text-[10px] text-paper placeholder:text-paper/20 focus:border-accent/50 focus:outline-none"
                      style={{ minWidth: (COL_WIDTHS[col] ?? 100) - 16 }}
                    />
                  </td>
                ))}
              </tr>
            )}
          </thead>
          <tbody>
            {filteredRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1} className="py-16 text-center text-paper/30 text-sm italic">
                  No rows match the current filters
                </td>
              </tr>
            ) : filteredRows.map((row, i) => (
              <tr
                key={i}
                className={`group border-b border-ink-line/30 transition-colors hover:bg-accent/5 ${i % 2 === 0 ? "bg-transparent" : "bg-ink/20"}`}
              >
                {/* Row number */}
                <td className="px-2 py-1.5 text-center text-[10px] text-paper/25 font-mono border-r border-ink-line/30 select-none group-hover:text-paper/50">
                  {i + 1}
                </td>
                {columns.map((col) => {
                  const val = String(row[col] ?? "");
                  const cellKey = `${i}-${col}`;
                  const isCopied = copiedCell === cellKey;
                  return (
                    <td
                      key={col}
                      className="px-3 py-1.5 border-r border-ink-line/30 cursor-pointer whitespace-nowrap max-w-xs group/cell relative"
                      style={{ maxWidth: COL_WIDTHS[col] ?? 100 }}
                      title={val}
                      onClick={() => copyCell(val, cellKey)}
                    >
                      {isCopied && (
                        <span className="absolute inset-0 flex items-center justify-center bg-accent/20 text-accent text-[10px] font-bold z-10 rounded">
                          Copied!
                        </span>
                      )}
                      {BADGE_COLS.has(col) ? (
                        <CellBadge col={col} value={val} />
                      ) : col === "plate_confidence" || col === "visibility" ? (
                        <span className="flex items-center gap-1.5">
                          <span className="font-mono text-paper/70">{val ? (parseFloat(val) * 100).toFixed(1) + "%" : "—"}</span>
                          {val && (
                            <span className="w-12 h-1.5 rounded-full bg-ink-line overflow-hidden">
                              <span
                                className="block h-full rounded-full bg-accent/60"
                                style={{ width: `${parseFloat(val) * 100}%` }}
                              />
                            </span>
                          )}
                        </span>
                      ) : col === "identity" ? (
                        <span className="font-mono text-accent/80 font-semibold">{val || "—"}</span>
                      ) : col === "plate_text" ? (
                        <span className="font-mono text-yellow-400/90 tracking-wider text-[11px]">{val || "—"}</span>
                      ) : col === "md5" ? (
                        <span className="font-mono text-[10px] text-paper/30 truncate block" style={{ maxWidth: 110 }}>{val}</span>
                      ) : (
                        <span className="font-mono text-paper/70 truncate block" style={{ maxWidth: COL_WIDTHS[col] ?? 100 }}>{val || <span className="text-paper/20">—</span>}</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Dataset Detail Panel ──────────────────────────────────────────────────

function DatasetDetail({ dataset }: { dataset: DatasetListEntry }) {
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const PAGE_SIZE = 200;

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDatasetRows(dataset.dataset_name, PAGE_SIZE, page * PAGE_SIZE)
      .then(r => setRows(r))
      .catch(e => setError(e.message ?? "Failed to load rows"))
      .finally(() => setLoading(false));
  }, [dataset.dataset_name, page]);

  const columns = rows.length > 0 && rows[0] ? Object.keys(rows[0]) : [];

  // Compute summary stats from visible rows
  const stats = useMemo(() => {
    if (!rows.length) return null;
    const identities = new Set(rows.map(r => r.identity).filter(Boolean));
    const cameras = new Set(rows.map(r => r.camera_id).filter(Boolean));
    const splits = new Set(rows.map(r => r.split).filter(Boolean));
    const plates = rows.filter(r => r.plate_text).length;
    return { identities: identities.size, cameras: cameras.size, splits: splits.size, plates };
  }, [rows]);

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Total Rows" value={dataset.row_count.toLocaleString()} />
        <StatCard label="CSV Size" value={fmtBytes(dataset.csv_size_bytes)} />
        <StatCard label="Created" value={fmtDate(dataset.created_at).split(",")[0] ?? ""} />
        <StatCard label="Previews" value={dataset.preview_count} accent="var(--color-accent)" />
      </div>

      {/* Computed stats from loaded rows */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Identities (page)" value={stats.identities} accent="#a78bfa" />
          <StatCard label="Cameras (page)" value={stats.cameras} accent="#34d399" />
          <StatCard label="Splits" value={stats.splits} />
          <StatCard label="With Plates (page)" value={stats.plates} accent="#fbbf24" />
        </div>
      )}

      {/* Download bar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-xs tracking-widest text-paper/50 uppercase">
            Data Table — Page {page + 1}
          </h3>
          <span className="text-[10px] font-mono text-accent/60">
            Rows {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, dataset.row_count)} of {dataset.row_count.toLocaleString()}
          </span>
        </div>
        <div className="flex gap-2">
          <a
            href={downloadDatasetCsv(dataset.dataset_name)}
            download
            id={`download-csv-${dataset.dataset_name}`}
            className="btn-primary px-4 py-1.5 text-xs flex items-center gap-1.5"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
            Download CSV
          </a>
        </div>
      </div>

      {/* Table area */}
      {loading ? (
        <div className="flex items-center gap-2 text-paper/50 text-sm py-8">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
          Loading rows…
        </div>
      ) : error ? (
        <div className="rounded-lg bg-red-900/10 border border-red-500/20 p-4 text-sm text-red-400">
          {error}
        </div>
      ) : rows.length > 0 ? (
        <>
          <ExcelTable rows={rows} columns={columns} />
          {/* Pagination */}
          <div className="flex items-center gap-3 pt-1">
            <button
              id="prev-page-btn"
              className="btn-primary px-4 py-1.5 text-xs disabled:opacity-40"
              disabled={page === 0}
              onClick={() => setPage(p => p - 1)}
            >← Prev</button>
            <span className="text-xs text-paper/50 font-mono">
              {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, dataset.row_count)} / {dataset.row_count.toLocaleString()}
            </span>
            <button
              id="next-page-btn"
              className="btn-primary px-4 py-1.5 text-xs disabled:opacity-40"
              disabled={(page + 1) * PAGE_SIZE >= dataset.row_count}
              onClick={() => setPage(p => p + 1)}
            >Next →</button>
          </div>
        </>
      ) : (
        <div className="text-sm text-paper/40 py-8 text-center italic">
          No rows returned from API — start the backend to view data.
        </div>
      )}
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<DatasetListEntry[]>([]);
  const [selected, setSelected] = useState<DatasetListEntry | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [tab, setTab] = useState<"browse" | "ingest">("browse");
  const [health, setHealth] = useState<BackendHealth | null>(null);

  const fetchDatasets = useCallback(async () => {
    setLoadingList(true);
    try {
      const list = await listDatasets();
      setDatasets(list);
      if (list.length > 0 && !selected) setSelected(list[0] ?? null);
    } catch {
      // API not running
    } finally {
      setLoadingList(false);
    }
  }, [selected]);

  useEffect(() => {
    checkBackendHealth().then(setHealth);
    fetchDatasets();
  }, []);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-[1600px] space-y-6">

          {/* Header */}
          <header className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="text-3xl font-light tracking-tight text-paper">Dataset Manager</h1>
              <p className="text-sm text-paper/60 mt-1">
                Browse processed CSV manifests. Click any cell to copy. Sort &amp; filter columns with the toolbar.
              </p>
            </div>
            <div className="flex gap-2 items-center">
              {/* Backend status pill */}
              {health && (
                <span className={`text-[10px] px-2 py-1 rounded-full border font-medium ${
                  health.ok ? "bg-green-500/10 border-green-500/30 text-green-400" : "bg-red-500/10 border-red-500/30 text-red-400"
                }`}>
                  {health.ok ? "● API Online" : "● API Offline"}
                </span>
              )}
              <button
                id="tab-browse"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  tab === "browse" ? "bg-accent/20 text-accent border border-accent/40" : "bg-ink text-paper/60 border border-ink-line hover:border-accent/30"
                }`}
                onClick={() => setTab("browse")}
              >Browse Datasets</button>
              <button
                id="tab-ingest"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  tab === "ingest" ? "bg-accent/20 text-accent border border-accent/40" : "bg-ink text-paper/60 border border-ink-line hover:border-accent/30"
                }`}
                onClick={() => setTab("ingest")}
              >+ Ingest New</button>
            </div>
          </header>

          {/* Offline banner */}
          {health && !health.ok && (
            <div className="rounded-xl border border-red-500/20 bg-red-900/10 px-5 py-3 text-sm text-red-400 flex items-center gap-3">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <span className="font-semibold">Backend offline.</span>{" "}
                <span className="text-red-400/70">{health.reason}</span>
              </div>
            </div>
          )}

          {tab === "ingest" ? (
            <IngestPanel onDone={() => { fetchDatasets(); setTab("browse"); }} />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
              {/* Sidebar */}
              <aside className="lg:col-span-1 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs tracking-widest text-paper/50 uppercase">Datasets</h2>
                  <button
                    id="refresh-datasets-btn"
                    onClick={fetchDatasets}
                    disabled={loadingList}
                    className="text-xs text-accent/70 hover:text-accent transition-colors disabled:opacity-50"
                  >
                    {loadingList ? "…" : "↻ Refresh"}
                  </button>
                </div>

                {datasets.length === 0 && !loadingList && (
                  <div className="text-xs text-paper/40 italic py-4 text-center">
                    No datasets processed yet.{" "}
                    <button className="text-accent underline" onClick={() => setTab("ingest")}>Ingest one</button>
                    {" "}or run{" "}
                    <code className="text-paper/60 text-[11px]">scripts/generate_veri_dataset.py</code>
                  </div>
                )}

                {datasets.map((ds) => (
                  <button
                    key={ds.dataset_name}
                    id={`dataset-btn-${ds.dataset_name}`}
                    onClick={() => setSelected(ds)}
                    className={`w-full text-left rounded-xl border p-4 transition-all ${
                      selected?.dataset_name === ds.dataset_name
                        ? "border-accent/60 bg-accent/10 shadow-[0_0_12px_rgba(0,240,255,0.1)]"
                        : "border-ink-line bg-ink hover:border-accent/30 hover:bg-ink-soft"
                    }`}
                  >
                    <div className="font-medium text-paper text-sm truncate">{ds.dataset_name}</div>
                    <div className="text-[10px] text-paper/50 mt-1">
                      {ds.row_count.toLocaleString()} rows · {fmtBytes(ds.csv_size_bytes)}
                    </div>
                    <div className="text-[10px] text-paper/40 font-mono">{ds.csv_file}</div>
                  </button>
                ))}

                {/* CLI reference */}
                <div className="mt-4 rounded-xl border border-ink-line bg-ink/50 p-4 space-y-2">
                  <h3 className="text-[10px] tracking-widest text-paper/40 uppercase">Quick Start</h3>
                  <pre className="text-[10px] text-accent/70 whitespace-pre-wrap break-all leading-relaxed">
{`# Generate 2,200-record dataset
python scripts/generate_veri_dataset.py

# Process ZIP
python scripts/process_dataset.py \\
  --zip dataset.zip \\
  --name veri776`}
                  </pre>
                </div>
              </aside>

              {/* Detail */}
              <section className="lg:col-span-4">
                {selected ? (
                  <div className="card p-6">
                    <div className="flex items-center justify-between mb-5">
                      <h2 className="text-xl font-light text-paper">
                        <span className="text-accent">/</span>{selected.dataset_name}
                      </h2>
                      <span className="text-xs text-paper/40 font-mono">{selected.csv_file}</span>
                    </div>
                    <DatasetDetail dataset={selected} />
                  </div>
                ) : (
                  <div className="card flex flex-col items-center justify-center py-24 text-center gap-4">
                    <div className="text-6xl text-paper/10">🗂</div>
                    <p className="text-sm text-paper/40">Select a dataset from the sidebar to explore its CSV manifest</p>
                    <button
                      id="ingest-first-dataset-btn"
                      className="btn-primary px-6 py-2 text-sm"
                      onClick={() => setTab("ingest")}
                    >
                      Ingest First Dataset
                    </button>
                  </div>
                )}
              </section>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

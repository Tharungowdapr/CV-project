"use client";

import { useState, useEffect, useRef } from "react";
import { Search, Filter, Crosshair, Upload, CheckCircle2, AlertTriangle, WifiOff, Database } from "lucide-react";
import { MatchConsole } from "@/components/features/MatchConsole";
import {
  searchByPlate,
  searchByPhoto,
  searchDatasets,
  checkBackendHealth,
  SightingResult,
  DatasetSearchResult,
  BackendHealth,
  ApiError,
} from "@/lib/api";
import { VerdictBadge } from "@/components/ui/VerdictBadge";

// ─── Types ──────────────────────────────────────────────────────────────────

type SearchMode = "plate" | "photo" | "csv";
type SearchStatus = "idle" | "searching" | "done" | "error";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function confidenceColor(c: number): string {
  if (c >= 0.8) return "text-green-400";
  if (c >= 0.6) return "text-yellow-400";
  return "text-red-400";
}

// ─── Error Banner ────────────────────────────────────────────────────────────

function ErrorBanner({ error, onDismiss }: { error: string; onDismiss: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-900/10 px-4 py-3 text-sm text-red-400">
      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
      <div className="flex-1">{error}</div>
      <button onClick={onDismiss} className="text-red-400/50 hover:text-red-400 transition-colors">✕</button>
    </div>
  );
}

// ─── Backend Status Banner ────────────────────────────────────────────────────

function BackendBanner({ health }: { health: BackendHealth }) {
  if (health.ok) return null;
  return (
    <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-900/10 px-4 py-3 text-sm text-amber-400">
      <WifiOff className="w-4 h-4 mt-0.5 flex-shrink-0" />
      <div>
        <span className="font-semibold">Backend API offline.</span>{" "}
        <span className="text-amber-400/70">
          {health.reason ?? "Start the server to enable photo/plate search."}
        </span>
        <br />
        <span className="text-amber-400/50 text-xs mt-1 block font-mono">
          uvicorn reiduq.serving.app:app --reload --port 8000
        </span>
        <span className="text-amber-400/70 text-xs mt-0.5 block">
          Dataset CSV search below is still available offline.
        </span>
      </div>
    </div>
  );
}

// ─── Gallery Sighting Card ───────────────────────────────────────────────────

function SightingRow({ item, idx }: { item: SightingResult; idx: number }) {
  return (
    <tr
      className={`border-b border-ink-line/50 transition-colors hover:bg-ink-soft/50 ${
        item.verdict === "UNCERTAIN" ? "bg-yellow-900/5" :
        item.verdict === "MATCH" ? "bg-green-900/5" : "bg-transparent"
      }`}
    >
      <td className="px-4 py-3 text-center text-[10px] text-paper/30 font-mono">{idx + 1}</td>
      <td className="px-4 py-3 font-mono text-paper/90 whitespace-nowrap">
        {item.camera_id}
        {item.verdict === "UNCERTAIN" && (
          <span className="block mt-1 text-[10px] text-yellow-400 font-bold">
            <Crosshair className="w-3 h-3 inline mr-1" /> HUMAN REVIEW
          </span>
        )}
      </td>
      <td className="px-4 py-3 text-paper/60 whitespace-nowrap text-xs">{item.dataset}</td>
      <td className="px-4 py-3 font-mono text-yellow-400 tracking-wider text-xs">
        {item.plate_text || <span className="text-paper/30">—</span>}
      </td>
      <td className="px-4 py-3 text-right whitespace-nowrap">
        <span className={`font-mono text-sm font-bold ${confidenceColor(item.confidence)}`}>
          {(item.confidence * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-4 py-3 text-center">
        <VerdictBadge verdict={item.verdict} />
      </td>
    </tr>
  );
}

// ─── CSV Search Result Row ──────────────────────────────────────────────────

function CsvResultRow({ item, idx, highlight }: { item: DatasetSearchResult; idx: number; highlight: string }) {
  const hl = (val: string) => {
    if (!highlight || !val) return val;
    const idx = val.toLowerCase().indexOf(highlight.toLowerCase());
    if (idx === -1) return val;
    return (
      <>
        {val.slice(0, idx)}
        <mark className="bg-accent/30 text-accent rounded px-0.5">{val.slice(idx, idx + highlight.length)}</mark>
        {val.slice(idx + highlight.length)}
      </>
    );
  };
  return (
    <tr className={`border-b border-ink-line/50 transition-colors hover:bg-ink-soft/50 ${idx % 2 === 0 ? "" : "bg-ink/20"}`}>
      <td className="px-3 py-2 text-center text-[10px] text-paper/30 font-mono">{idx + 1}</td>
      <td className="px-3 py-2">
        <span className="font-mono text-accent/80 font-semibold text-xs">{hl(item.identity ?? "—")}</span>
      </td>
      <td className="px-3 py-2">
        <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium border bg-blue-500/20 text-blue-300 border-blue-500/30">
          {item.camera_id ?? "—"}
        </span>
      </td>
      <td className="px-3 py-2 font-mono text-yellow-400 tracking-wider text-xs">
        {hl(item.plate_text ?? "") || <span className="text-paper/30">—</span>}
      </td>
      <td className="px-3 py-2 text-xs text-paper/60">{hl(item.vehicle_type ?? "")}</td>
      <td className="px-3 py-2 text-xs">
        {item.color && (
          <span className="flex items-center gap-1 text-paper/70">
            <span className="w-2.5 h-2.5 rounded-full border border-ink-line inline-block" style={{
              background: {
                black: "#111", white: "#eee", silver: "#aaa", red: "#e44", blue: "#44e",
                green: "#4a4", yellow: "#ee4", gray: "#888", brown: "#964", gold: "#ca4",
              }[item.color ?? ""] ?? "#666",
            }} />
            {item.color}
          </span>
        )}
      </td>
      <td className="px-3 py-2 text-xs text-paper/50">{item.brand ?? "—"}</td>
      <td className="px-3 py-2">
        {item.split && (
          <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] border ${
            item.split === "train" ? "bg-blue-500/20 text-blue-300 border-blue-500/30" :
            item.split === "query" ? "bg-purple-500/20 text-purple-300 border-purple-500/30" :
            item.split === "gallery" ? "bg-green-500/20 text-green-300 border-green-500/30" :
            "bg-accent/20 text-accent border-accent/30"
          }`}>{item.split}</span>
        )}
      </td>
      <td className="px-3 py-2 text-[10px] font-mono text-paper/40 max-w-[180px] truncate">{item.dataset_name}</td>
    </tr>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function SearchPage() {
  // Query state
  const [plateQuery, setPlateQuery] = useState("");
  const [color, setColor] = useState("Any Color");
  const [vehicleType, setVehicleType] = useState("Any Type");
  const [splitFilter, setSplitFilter] = useState("all");
  const [searchField, setSearchField] = useState("all");
  const [selectedPhoto, setSelectedPhoto] = useState<File | null>(null);
  const [mode, setMode] = useState<SearchMode>("plate");

  // Results state
  const [sightingResults, setSightingResults] = useState<SightingResult[]>([]);
  const [csvResults, setCsvResults] = useState<DatasetSearchResult[]>([]);
  const [csvPage, setCsvPage] = useState(0);
  const CSV_PAGE_SIZE = 100;

  // UI state
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [searchError, setSearchError] = useState<string | null>(null);
  const [errorKind, setErrorKind] = useState<string | null>(null);
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(true);

  // Polling / abort
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    checkBackendHealth().then(h => { setHealth(h); setCheckingHealth(false); });
  }, []);

  // Auto-switch to CSV mode if backend offline
  useEffect(() => {
    if (health && !health.ok && mode !== "csv") setMode("csv");
  }, [health]);

  async function handleSearch() {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setStatus("searching");
    setSearchError(null);
    setErrorKind(null);
    setSightingResults([]);
    setCsvResults([]);
    setCsvPage(0);

    try {
      if (mode === "csv") {
        // CSV-based search — works without backend gallery index
        const q = plateQuery.trim() || "";
        const results = await searchDatasets(q, searchField, splitFilter === "all" ? "" : splitFilter, 500, 0);
        setCsvResults(results);
        setStatus("done");
      } else if (mode === "photo" && selectedPhoto) {
        const results = await searchByPhoto(selectedPhoto, 20);
        setSightingResults(results);
        setStatus(results.length === 0 ? "done" : "done");
        if (results.length === 0) {
          setSearchError("No matches found. The gallery index may not be built yet — try Dataset CSV search.");
          setErrorKind("not_ready");
        }
      } else if (mode === "plate") {
        if (!plateQuery.trim()) {
          setSearchError("Please enter a license plate or switch to CSV search.");
          setStatus("error");
          return;
        }
        const results = await searchByPlate(plateQuery.trim(), 20);
        setSightingResults(results);
        setStatus("done");
        if (results.length === 0) {
          setSearchError("No matches found. Try CSV search to query the processed dataset directly.");
          setErrorKind("not_ready");
        }
      } else {
        setSearchError("Please enter a plate query or upload a photo.");
        setStatus("error");
      }
    } catch (e) {
      const err = e as ApiError;
      setStatus("error");
      if (err.kind === "network") {
        setSearchError("Cannot reach backend server. Switching to Dataset CSV search.");
        setErrorKind("network");
        setMode("csv");
        // Auto-retry with CSV
        try {
          const results = await searchDatasets(plateQuery.trim(), searchField, splitFilter === "all" ? "" : splitFilter, 500, 0);
          setCsvResults(results);
          setStatus("done");
          setSearchError(null);
        } catch {}
      } else if (err.kind === "not_ready") {
        setSearchError("Gallery index not built yet. Use Dataset CSV search below, or run scripts/build_gallery_index.py.");
        setErrorKind("not_ready");
      } else {
        setSearchError(err.message ?? "Search failed. Please try again.");
        setErrorKind(err.kind ?? "unknown");
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const csvPaged = csvResults.slice(csvPage * CSV_PAGE_SIZE, (csvPage + 1) * CSV_PAGE_SIZE);
  const csvTotalPages = Math.ceil(csvResults.length / CSV_PAGE_SIZE);

  return (
    <div className="space-y-6 animate-in fade-in duration-700">
      {/* Page header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-accent">
            GLOBAL SEARCH
          </h1>
          <p className="text-accent/60 mt-1 text-sm">
            Query by plate, photo, or search the processed dataset CSV directly.
          </p>
        </div>
        {/* Health badge */}
        {!checkingHealth && health && (
          <span className={`text-[10px] px-3 py-1.5 rounded-full border font-medium flex items-center gap-1.5 ${
            health.ok
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : "bg-amber-500/10 border-amber-500/30 text-amber-400"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${health.ok ? "bg-green-400 animate-pulse" : "bg-amber-400"}`} />
            {health.ok ? "API Online" : "API Offline — CSV mode"}
          </span>
        )}
      </div>

      {/* Backend warning */}
      {health && !health.ok && <BackendBanner health={health} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Left panel: Query controls ── */}
        <div className="lg:col-span-1 space-y-5">

          {/* Mode tabs */}
          <div className="card p-5">
            <h3 className="text-xs tracking-widest text-paper/50 uppercase mb-3">Search Mode</h3>
            <div className="grid grid-cols-3 gap-1.5 rounded-lg bg-ink p-1 border border-ink-line">
              {(["plate", "photo", "csv"] as SearchMode[]).map(m => (
                <button
                  key={m}
                  id={`mode-${m}`}
                  onClick={() => setMode(m)}
                  disabled={m !== "csv" && health?.ok === false}
                  className={`py-2 px-2 rounded-md text-[10px] font-semibold tracking-widest uppercase transition-all ${
                    mode === m
                      ? "bg-accent/20 text-accent border border-accent/40 shadow-[0_0_8px_rgba(0,240,255,0.15)]"
                      : "text-paper/40 hover:text-paper/70 disabled:opacity-30 disabled:cursor-not-allowed"
                  }`}
                >
                  {m === "plate" ? "🪪 Plate" : m === "photo" ? "📷 Photo" : "📊 CSV"}
                </button>
              ))}
            </div>
            {mode !== "csv" && health?.ok === false && (
              <p className="text-[10px] text-amber-400/70 mt-2">Plate/Photo search requires the backend API.</p>
            )}
          </div>

          {/* Query parameters */}
          <div className="card p-5 border-accent/20">
            <h3 className="text-xs tracking-widest text-paper/50 uppercase mb-4">Query Parameters</h3>
            <div className="space-y-4">
              {/* Plate input (for plate + csv modes) */}
              {(mode === "plate" || mode === "csv") && (
                <div>
                  <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">
                    {mode === "csv" ? "Search Text" : "License Plate (OCR)"}
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-2.5 w-4 h-4 text-accent/40" />
                    <input
                      id="search-input"
                      type="text"
                      className="field pl-9 font-mono"
                      placeholder={mode === "csv" ? "e.g. MH-01 or sedan or blue…" : "e.g. MH-01-AB-1234"}
                      value={plateQuery}
                      onChange={e => setPlateQuery(e.target.value)}
                      onKeyDown={handleKeyDown}
                    />
                  </div>
                </div>
              )}

              {/* Photo upload */}
              {mode === "photo" && (
                <div>
                  <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">Upload Query Photo</label>
                  <input
                    id="photo-upload"
                    type="file"
                    accept="image/*"
                    onChange={e => setSelectedPhoto(e.target.files?.[0] ?? null)}
                    className="field text-xs file:mr-3 file:py-1 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-accent/20 file:text-accent hover:file:bg-accent/30 cursor-pointer"
                  />
                  {selectedPhoto && (
                    <p className="text-xs text-accent mt-1.5 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> {selectedPhoto.name}
                    </p>
                  )}
                </div>
              )}

              {/* CSV-specific filters */}
              {mode === "csv" && (
                <>
                  <div>
                    <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">Search Field</label>
                    <select
                      id="search-field"
                      className="field appearance-none"
                      value={searchField}
                      onChange={e => setSearchField(e.target.value)}
                    >
                      <option value="all">All Columns</option>
                      <option value="plate_text">Plate Text</option>
                      <option value="identity">Identity / Vehicle ID</option>
                      <option value="camera_id">Camera ID</option>
                      <option value="vehicle_type">Vehicle Type</option>
                      <option value="color">Color</option>
                      <option value="brand">Brand</option>
                      <option value="split">Split</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">Split Filter</label>
                    <select
                      id="split-filter"
                      className="field appearance-none"
                      value={splitFilter}
                      onChange={e => setSplitFilter(e.target.value)}
                    >
                      <option value="all">All Splits</option>
                      <option value="train">Train</option>
                      <option value="query">Query</option>
                      <option value="gallery">Gallery</option>
                    </select>
                  </div>
                </>
              )}

              {/* Shared attribute filters */}
              {mode !== "csv" && (
                <>
                  <div>
                    <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">Vehicle Color</label>
                    <select id="color-filter" className="field appearance-none" value={color} onChange={e => setColor(e.target.value)}>
                      <option>Any Color</option>
                      <option>Black</option><option>White</option><option>Silver / Gray</option>
                      <option>Red</option><option>Blue</option><option>Green</option><option>Yellow</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] tracking-wider text-accent/60 uppercase block mb-1">Vehicle Type</label>
                    <select id="type-filter" className="field appearance-none" value={vehicleType} onChange={e => setVehicleType(e.target.value)}>
                      <option>Any Type</option>
                      <option>Sedan</option><option>SUV</option><option>Truck</option><option>Van</option><option>Bus</option>
                    </select>
                  </div>
                </>
              )}

              {/* Error display */}
              {searchError && status !== "searching" && (
                <ErrorBanner error={searchError} onDismiss={() => { setSearchError(null); setErrorKind(null); }} />
              )}

              {/* Search button */}
              <div className="pt-1">
                <button
                  id="execute-search-btn"
                  onClick={handleSearch}
                  disabled={status === "searching"}
                  className="btn-primary w-full py-3 disabled:opacity-50 relative overflow-hidden"
                >
                  {status === "searching" ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Searching…
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <Search className="w-4 h-4" />
                      {mode === "csv" ? "Search Dataset" : "Execute Query"}
                    </span>
                  )}
                </button>
                {mode === "csv" && (
                  <p className="text-[10px] text-paper/30 text-center mt-1.5">
                    Press Enter or click — searches all processed CSVs
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Quick stats if CSV results */}
          {csvResults.length > 0 && mode === "csv" && (
            <div className="card p-4 space-y-3">
              <h3 className="text-xs tracking-widest text-paper/50 uppercase">Result Summary</h3>
              <div className="space-y-2 text-xs">
                {[
                  ["Total matches", csvResults.length.toLocaleString()],
                  ["Unique vehicles", new Set(csvResults.map(r => r.identity)).size.toLocaleString()],
                  ["Unique cameras", new Set(csvResults.map(r => r.camera_id)).size.toLocaleString()],
                  ["With plates", csvResults.filter(r => r.plate_text).length.toLocaleString()],
                ].map(([label, val]) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-paper/50">{label}</span>
                    <span className="font-mono text-accent">{val}</span>
                  </div>
                ))}
              </div>
              <button
                id="export-csv-btn"
                onClick={() => {
                  const headers = Object.keys(csvResults[0] ?? {}).join(",");
                  const rows = csvResults.map(r => Object.values(r).map(v => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
                  const blob = new Blob([headers + "\n" + rows], { type: "text/csv" });
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(blob);
                  a.download = `search_results_${Date.now()}.csv`;
                  a.click();
                }}
                className="w-full text-center text-xs btn-primary py-1.5"
              >
                ⬇ Export Results CSV
              </button>
            </div>
          )}
        </div>

        {/* ── Right panel: Results ── */}
        <div className="lg:col-span-2 space-y-5">

          {/* Gallery / FAISS results */}
          {sightingResults.length > 0 && (
            <div className="card p-5 border-accent/30 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs tracking-widest text-accent uppercase">
                  Gallery Matches ({sightingResults.length} sightings)
                </h3>
                <button
                  id="export-sightings-btn"
                  className="text-xs btn-primary px-3 py-1"
                  onClick={() => {
                    const csv = ["Location,Dataset,Plate,Confidence,Verdict,Reason"];
                    sightingResults.forEach(r => csv.push(`${r.camera_id},${r.dataset},${r.plate_text || ""},${r.confidence.toFixed(3)},${r.verdict},"${r.reason}"`));
                    const blob = new Blob([csv.join("\n")], { type: "text/csv" });
                    const a = document.createElement("a");
                    a.href = URL.createObjectURL(blob);
                    a.download = "search_results.csv";
                    a.click();
                  }}
                >⬇ Export CSV</button>
              </div>
              <div className="overflow-x-auto rounded-xl border border-ink-line max-h-[500px]">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 bg-ink z-10 border-b border-ink-line">
                    <tr>
                      {["#", "Camera / Location", "Dataset", "Plate (OCR)", "Confidence", "Verdict"].map(h => (
                        <th key={h} className="px-4 py-3 text-left font-medium tracking-wider text-paper/50 uppercase">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sightingResults.map((item, idx) => <SightingRow key={idx} item={item} idx={idx} />)}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* CSV Dataset search results */}
          {csvResults.length > 0 && (
            <div className="card p-5 border-purple-500/20 space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-purple-400" />
                  <h3 className="text-xs tracking-widest text-purple-400 uppercase">
                    Dataset CSV Results — {csvResults.length.toLocaleString()} matches
                  </h3>
                </div>
                <div className="flex items-center gap-2">
                  {csvTotalPages > 1 && (
                    <span className="text-[10px] text-paper/40 font-mono">
                      Page {csvPage + 1}/{csvTotalPages}
                    </span>
                  )}
                </div>
              </div>

              <div className="overflow-x-auto rounded-xl border border-ink-line max-h-[520px]">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 z-10 border-b border-ink-line" style={{ background: "hsl(220,20%,12%)" }}>
                    <tr>
                      {["#", "Identity", "Camera", "Plate", "Type", "Color", "Brand", "Split", "Dataset"].map(h => (
                        <th key={h} className="px-3 py-2 text-left font-medium tracking-wider text-paper/50 uppercase whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvPaged.map((item, idx) => (
                      <CsvResultRow key={idx} item={item} idx={csvPage * CSV_PAGE_SIZE + idx} highlight={plateQuery} />
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {csvTotalPages > 1 && (
                <div className="flex items-center gap-3">
                  <button
                    id="csv-prev-page"
                    className="btn-primary px-4 py-1.5 text-xs disabled:opacity-40"
                    disabled={csvPage === 0}
                    onClick={() => setCsvPage(p => Math.max(0, p - 1))}
                  >← Prev</button>
                  <span className="text-xs text-paper/50 font-mono">
                    {csvPage * CSV_PAGE_SIZE + 1}–{Math.min((csvPage + 1) * CSV_PAGE_SIZE, csvResults.length)} of {csvResults.length.toLocaleString()}
                  </span>
                  <button
                    id="csv-next-page"
                    className="btn-primary px-4 py-1.5 text-xs disabled:opacity-40"
                    disabled={csvPage >= csvTotalPages - 1}
                    onClick={() => setCsvPage(p => Math.min(csvTotalPages - 1, p + 1))}
                  >Next →</button>
                </div>
              )}
            </div>
          )}

          {/* Idle / no results states */}
          {status === "idle" && sightingResults.length === 0 && csvResults.length === 0 && (
            <div className="card p-6 border-accent/20">
              <h3 className="text-xs tracking-widest text-paper/50 uppercase mb-4">Manual Score Console (OSNet-AIN)</h3>
              <MatchConsole />
            </div>
          )}

          {status === "done" && sightingResults.length === 0 && csvResults.length === 0 && !searchError && (
            <div className="card p-10 flex flex-col items-center justify-center gap-4 text-center border-dashed border-ink-line">
              <div className="text-5xl">🔍</div>
              <p className="text-paper/40 text-sm">No results found for this query.</p>
              <p className="text-paper/30 text-xs">Try switching to CSV mode to search the full dataset.</p>
              <button
                id="switch-to-csv"
                onClick={() => setMode("csv")}
                className="btn-primary px-5 py-2 text-xs mt-1"
              >
                Switch to CSV Search
              </button>
            </div>
          )}

          {/* Error state with helpful actions */}
          {status === "error" && (
            <div className="card p-6 border-red-500/20 bg-red-900/5 space-y-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-red-400 mb-1">Search Failed</h3>
                  <p className="text-xs text-red-400/70">{searchError}</p>
                </div>
              </div>
              <div className="flex gap-3 flex-wrap">
                <button
                  id="retry-as-csv"
                  onClick={() => { setMode("csv"); handleSearch(); }}
                  className="btn-primary px-4 py-2 text-xs flex items-center gap-1.5 bg-purple-900/30 border-purple-500/30 text-purple-300"
                >
                  <Database className="w-3 h-3" /> Try CSV Search
                </button>
                <button
                  id="dismiss-error"
                  onClick={() => { setStatus("idle"); setSearchError(null); }}
                  className="btn-primary px-4 py-2 text-xs bg-ink border-ink-line text-paper/50"
                >
                  Dismiss
                </button>
              </div>

              {errorKind === "not_ready" && (
                <div className="rounded-lg bg-ink p-3 border border-ink-line text-[11px] text-paper/50 font-mono space-y-1">
                  <div className="text-paper/70 font-semibold mb-1">To build the gallery index:</div>
                  <div>python scripts/build_gallery_index.py</div>
                  <div className="text-paper/30"># or use the Dataset CSV search above</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

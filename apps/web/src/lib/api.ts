import type { HealthResponse, LoginResponse, MatchRequest, MatchResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface SightingResult {
  id?: number;
  image_path: string;
  camera_id: string;
  dataset: string;
  similarity: number;
  confidence: number;
  verdict: "MATCH" | "UNCERTAIN" | "REJECT";
  reason: string;
  plate_text?: string | null;
  plate_confidence?: number | null;
  visibility?: number | null;
  indexed_at?: string | null;
}

export interface GalleryEntry {
  faiss_position: number;
  uid: string;
  image_path: string;
  dataset: string;
  camera_id: string;
  ground_truth_identity?: number | null;
  plate_text?: string | null;
  plate_confidence?: number | null;
  visibility?: number | null;
  indexed_at?: string;
}

export interface IngestionStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  filename: string;
  total_frames?: number;
  processed_frames?: number;
  detections?: number;
  embeddings?: number;
  elapsed_seconds?: number;
  eta_seconds?: number;
  error?: string;
}

// ─── Error types ────────────────────────────────────────────────────────────

export type ApiErrorKind = "network" | "not_ready" | "not_found" | "server" | "unknown";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly kind: ApiErrorKind,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ─── Core request helper ────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init });
  } catch {
    throw new ApiError(
      "Cannot connect to backend. Make sure the API server is running.",
      "network",
    );
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string; detail?: string };
    const msg = body.error ?? body.detail ?? `request failed (${res.status})`;
    const kind: ApiErrorKind =
      res.status === 503 ? "not_ready" :
      res.status === 404 ? "not_found" :
      res.status >= 500 ? "server" : "unknown";
    throw new ApiError(msg, kind, res.status);
  }
  return (await res.json()) as T;
}

// ─── Backend health ─────────────────────────────────────────────────────────

export interface BackendHealth {
  ok: boolean;
  reason?: string;
  status?: string;
}

export async function checkBackendHealth(): Promise<BackendHealth> {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return { ok: false, reason: `health check failed (${r.status})` };
    const body = await r.json();
    return { ok: true, status: body.status };
  } catch {
    return { ok: false, reason: "Backend unreachable — start with: uvicorn reiduq.serving.app:app --reload" };
  }
}

export const getHealth = (): Promise<HealthResponse> =>
  fetch(`${API_BASE}/health`)
    .then((r) => {
      if (!r.ok) throw new Error(`health check failed (${r.status})`);
      return r.json() as Promise<HealthResponse>;
    });

export const login = (apiKey: string) =>
  request<LoginResponse>("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });

export const logout = () => request<{ status: string }>("/v1/auth/logout", { method: "POST" });

export const postMatch = (payload: MatchRequest) =>
  request<MatchResponse>("/v1/match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const searchByPlate = (plateQuery: string, k: number = 10) =>
  request<SightingResult[]>(
    `/v1/search/plate?plate_query=${encodeURIComponent(plateQuery)}&k=${k}`
  );

export const searchByPhoto = async (imageFile: File, k: number = 10): Promise<SightingResult[]> => {
  const formData = new FormData();
  formData.append("image", imageFile);
  return request<SightingResult[]>(`/v1/search/photo?k=${k}`, {
    method: "POST",
    body: formData,
  });
};

export const getGalleryEntries = (limit: number = 200, offset: number = 0) =>
  request<GalleryEntry[]>(`/v1/gallery?limit=${limit}&offset=${offset}`);

export const getGalleryStats = () =>
  request<{ total: number; cameras: number; datasets: number; plates: number }>("/v1/gallery/stats");

export const deleteDataset = (datasetName: string) =>
  request<{ status: string; deleted: number }>(`/v1/gallery/dataset/${encodeURIComponent(datasetName)}`, {
    method: "DELETE",
  });

export const configureLiveStream = (streamUrl: string) =>
  request<{ status: string; stream_url: string }>("/api/live/camera/cam_01/config", {
    method: "POST",
    body: JSON.stringify({ stream_url: streamUrl }),
  });

export const ingestVideo = async (
  file: File,
  onProgress?: (status: IngestionStatus) => void
): Promise<IngestionStatus> => {
  const formData = new FormData();
  formData.append("video", file);
  const result = await request<IngestionStatus>("/v1/ingest/video", {
    method: "POST",
    body: formData,
  });
  return result;
};

export const getIngestionStatus = (jobId: string) =>
  request<IngestionStatus>(`/v1/ingest/status/${jobId}`);

// ─────────────────────────────────────────────────────────────────
//  Dataset Ingest (ZIP / folder → CSV manifest)
// ─────────────────────────────────────────────────────────────────

export interface DatasetIngestStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  dataset_name: string;
  total_images: number;
  processed: number;
  preview_count: number;
  csv_path: string | null;
  errors: string[];
  elapsed_seconds: number;
  error?: string | null;
}

export interface DatasetListEntry {
  dataset_name: string;
  csv_file: string;
  row_count: number;
  preview_count: number;
  created_at: string;
  csv_size_bytes: number;
}

export interface DatasetPreview {
  filename: string;
  path: string;
}

export const ingestDatasetZip = async (
  file: File,
  datasetName: string,
  split: string = "all",
  nPreview: number = 20,
): Promise<DatasetIngestStatus> => {
  const formData = new FormData();
  formData.append("file", file);
  const params = new URLSearchParams({
    dataset_name: datasetName,
    split,
    n_preview: nPreview.toString(),
  });
  return request<DatasetIngestStatus>(`/v1/dataset/ingest?${params}`, {
    method: "POST",
    body: formData,
  });
};

export const getDatasetIngestStatus = (jobId: string) =>
  request<DatasetIngestStatus>(`/v1/dataset/ingest/status/${jobId}`);

export const listDatasets = () =>
  request<DatasetListEntry[]>("/v1/dataset/list");

export const getDatasetRows = (
  datasetName: string,
  limit: number = 100,
  offset: number = 0,
) =>
  request<Record<string, string>[]>(
    `/v1/dataset/${encodeURIComponent(datasetName)}/rows?limit=${limit}&offset=${offset}`
  );

export const getDatasetPreviews = (datasetName: string) =>
  request<DatasetPreview[]>(`/v1/dataset/${encodeURIComponent(datasetName)}/previews`);

export const downloadDatasetCsv = (datasetName: string): string =>
  `${API_BASE}/v1/dataset/${encodeURIComponent(datasetName)}/csv`;

// ─────────────────────────────────────────────────────────────────
//  Dataset search (CSV-based, works without FAISS gallery index)
// ─────────────────────────────────────────────────────────────────

export interface DatasetSearchResult {
  uid: string;
  dataset_name: string;
  split: string;
  identity?: string;
  camera_id?: string;
  plate_text?: string;
  vehicle_type?: string;
  color?: string;
  brand?: string;
  confidence?: string;
  visibility?: string;
  [key: string]: string | undefined;
}

export const searchDatasets = (
  q: string,
  field: string = "all",
  split: string = "",
  limit: number = 100,
  offset: number = 0,
) =>
  request<DatasetSearchResult[]>(
    `/v1/dataset/search?q=${encodeURIComponent(q)}&field=${field}&split=${encodeURIComponent(split)}&limit=${limit}&offset=${offset}`
  );

export const searchDatasetsCount = (q: string, field: string = "all", split: string = "") =>
  request<{ count: number; query: string; field: string; split: string }>(
    `/v1/dataset/search/count?q=${encodeURIComponent(q)}&field=${field}&split=${encodeURIComponent(split)}`
  );

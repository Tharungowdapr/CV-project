export type Verdict = "MATCH" | "UNCERTAIN" | "REJECT";

export interface MatchRequest {
  query_id: string;
  similarities: number[];
  candidate_ids: string[];
  visibility: number;
  part_visibility?: number[];
  n_frames: number;
}

export interface MatchResponse {
  verdict: Verdict;
  matched_id: string | null;
  confidence: number;
  temperature: number;
  reason: string;
  model_version: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  model_loaded: boolean;
  calibrator: string;
}

export interface LoginResponse {
  role: string;
  expires_in_seconds: number;
}

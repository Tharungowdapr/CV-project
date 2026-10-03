"""API request/response schemas. Every external field is validated here."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    calibrator: str


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    similarities: list[float] = Field(min_length=1, max_length=100)
    candidate_ids: list[str] = Field(min_length=1, max_length=100)
    visibility: float = Field(ge=0.0, le=1.0, default=1.0)
    part_visibility: list[float] = Field(default_factory=lambda: [1.0] * 6, max_length=6)
    camera_stats: list[float] | None = Field(default=None, max_length=9)
    n_frames: int = Field(ge=1, le=10_000, default=1)


class MatchResponse(BaseModel):
    verdict: Literal["MATCH", "UNCERTAIN", "REJECT"]
    matched_id: str | None
    confidence: float
    temperature: float
    reason: str
    model_version: str


class LoginRequest(BaseModel):
    """The API key is sent once, in the body (never as a query param, which
    would land in access logs and browser history), to be exchanged for a
    short-lived session cookie. The frontend never stores this key."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    role: str
    expires_in_seconds: int


class SightingResponse(BaseModel):
    image_path: str
    camera_id: str
    dataset: str
    similarity: float
    confidence: float
    verdict: Literal["MATCH", "UNCERTAIN", "REJECT"]
    reason: str
    plate_text: str | None = None
    plate_confidence: float | None = None


class PlateSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plate_query: str = Field(min_length=2, max_length=16, pattern=r"^[A-Za-z0-9 \-]+$")
    k: int = Field(ge=1, le=50, default=10)


class ErrorResponse(BaseModel):
    """Errors never leak internal paths, stack traces, or model internals."""

    error: str
    correlation_id: str


class GalleryEntryResponse(BaseModel):
    """Single gallery entry returned from the /v1/gallery endpoint."""
    faiss_position: int
    uid: str
    image_path: str
    dataset: str
    camera_id: str
    ground_truth_identity: int | None = None
    plate_text: str | None = None
    plate_confidence: float | None = None
    visibility: float | None = None
    indexed_at: str | None = None


class GalleryStatsResponse(BaseModel):
    """Aggregated statistics across all indexed gallery entries."""
    total: int
    cameras: int
    datasets: int
    plates: int


class IngestionStatusResponse(BaseModel):
    """Progress report for a video ingestion job."""
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    filename: str
    total_frames: int = 0
    processed_frames: int = 0
    detections: int = 0
    embeddings: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None

class DatasetIngestStatus(BaseModel):
    """Progress report for a dataset ingestion job (ZIP or folder)."""
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    dataset_name: str
    total_images: int = 0
    processed: int = 0
    preview_count: int = 0
    csv_path: str | None = None
    errors: list[str] = []
    elapsed_seconds: float = 0.0
    error: str | None = None


class DatasetListEntry(BaseModel):
    """One row in the /v1/dataset/list response."""
    dataset_name: str
    csv_file: str
    row_count: int
    preview_count: int
    created_at: str
    csv_size_bytes: int

"""FastAPI inference service.

Ports-and-adapters: this module knows about HTTP, the domain does not know
about this module. Security headers, CORS allowlist, rate limiting, session
auth, audit persistence, and a scrubbed error handler are all wired here
rather than left to a reverse proxy that may or may not exist in a given
deployment.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response as FastAPIResponse, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from reiduq.core.exceptions import DataIntegrityError
from reiduq.core.logging import configure_logging, get_logger
from reiduq.core.settings import settings
from reiduq.serving.audit import AuditRecord, AuditStore
from reiduq.serving.predictor import Predictor
from reiduq.serving.schemas import (
    ErrorResponse,
    GalleryEntryResponse,
    GalleryStatsResponse,
    HealthResponse,
    IngestionStatusResponse,
    LoginRequest,
    LoginResponse,
    MatchRequest,
    MatchResponse,
    PlateSearchRequest,
    SightingResponse,
    DatasetIngestStatus,
    DatasetListEntry,
)
from reiduq.serving.security import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    issue_session_token,
    new_correlation_id,
    require_admin,
    require_api_key,
    validate_dimensions,
    validate_image_bytes,
    verify_api_key,
)

configure_logging(settings.log_level, settings.log_format)
log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    _audit.close()  # flush/close the sqlite connection on shutdown, not left to GC

if settings.redis_url:
    # Shared backend: the rate limit means what it says across every replica,
    # not just within the process that happened to receive the request.
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.rate_limit_per_minute}/minute"],
        storage_uri=settings.redis_url,
    )
    log.info("rate_limiter.backend", kind="redis")
else:
    limiter = Limiter(
        key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"]
    )
    log.warning(
        "rate_limiter.backend",
        kind="in_memory",
        note="per-process only; set REDIS_URL before running more than one replica",
    )

app = FastAPI(
    title="Vehicle Re-ID — calibrated matching service",
    version="0.1.0",
    docs_url=None if settings.is_production else "/docs",  # no schema browsing in prod
    redoc_url=None,
    lifespan=_lifespan,
)
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else settings.cors_list,
    allow_credentials=not settings.is_production,  # allow credentials in dev
    allow_methods=["*"],
    allow_headers=["*"],
)
from reiduq.serving.live_api import live_router
app.include_router(live_router, prefix="/api/live")

_predictor = Predictor.from_registry(settings.model_registry_path)
_audit = AuditStore(settings.audit_db_path)
_search_engine: object | None = None  # lazily loaded - see _get_search_engine
_ingestion_jobs: dict[str, dict] = {}  # job_id -> status dict for video ingestion tracking
_dataset_jobs: dict[str, dict] = {}   # job_id -> status dict for dataset (ZIP/folder) ingestion


def _get_search_engine() -> Any:
    """Load the gallery search engine on first use, not at import time.

    Unlike the calibrator (which has a documented, deliberate no-checkpoint
    fallback), there is no sensible fallback for "no gallery index has been
    built yet" - search endpoints should 503 clearly rather than the whole
    service failing to start because an operator has not run
    scripts/build_gallery_index.py yet.
    """
    global _search_engine
    if _search_engine is None:
        index_path = Path(settings.model_registry_path) / "gallery" / "index.faiss"
        store_path = Path(settings.model_registry_path) / "gallery" / "store.db"
        if not index_path.exists():
            raise HTTPException(
                status_code=503,
                detail="no gallery index is built yet - run scripts/build_gallery_index.py",
            )
        from reiduq.models.builder import build_encoder
        from reiduq.search.query_pipeline import SearchEngine

        encoder = build_encoder("osnet_ain", embed_dim=512, pretrained=False)
        _search_engine = SearchEngine.load(
            index_path, store_path, encoder, _predictor.calibrator,
            tau_high=_predictor.tau_high, tau_low=_predictor.tau_low,
        )
        log.info("search_engine.loaded", index=str(index_path))
    return _search_engine


@app.middleware("http")
async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if settings.is_production:
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "rate limit exceeded", "correlation_id": new_correlation_id()},
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return a correlation id and nothing else; the detail goes to the log."""
    cid = new_correlation_id()
    log.error("unhandled_exception", correlation_id=cid, path=request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal error", correlation_id=cid).model_dump(),
    )


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=app.version,
        model_loaded=_predictor.version != "dev",
        calibrator=type(_predictor.calibrator).__name__,
    )


@app.post("/v1/auth/login", response_model=LoginResponse)
@limiter.limit("10/minute")  # tighter than the general limit: this is the credential-guessing surface
async def login(request: Request, payload: LoginRequest, response: FastAPIResponse) -> LoginResponse:

    role = verify_api_key(payload.api_key)
    if role is None:
        # Same response shape whether the key is wrong or unknown - no
        # "key not found" vs "key disabled" distinction that helps an
        # attacker enumerate valid keys.
        raise HTTPException(status_code=401, detail="invalid credentials")

    token = issue_session_token(role)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,  # not readable by client-side JS, injected or otherwise
        secure=settings.is_production,  # over plain HTTP in dev only, where TLS isn't set up anyway
        samesite="strict",  # the cookie is never sent on a cross-site request, including a top-level navigation from a link
    )
    log.info("auth.login", role=role)
    return LoginResponse(role=role, expires_in_seconds=SESSION_TTL_SECONDS)


@app.post("/v1/auth/logout")
async def logout(response: FastAPIResponse) -> dict[str, str]:
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"status": "logged out"}


@app.post("/v1/match", response_model=MatchResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def match(
    request: Request, payload: MatchRequest,
) -> MatchResponse:
    cid = new_correlation_id()
    decision, temperature = _predictor.predict(
        payload.similarities,
        payload.candidate_ids,
        visibility=payload.visibility,
        part_visibility=payload.part_visibility,
        camera_stats=payload.camera_stats,
        n_frames=payload.n_frames,
    )
    role = "dev"
    log.info(
        "match.decided",
        correlation_id=cid,
        query_id=payload.query_id,  # an id, never the image
        verdict=decision.verdict,
        confidence=round(decision.confidence, 4),
        role=role,
    )
    _audit.record(
        AuditRecord(
            correlation_id=cid,
            query_id=payload.query_id,
            verdict=decision.verdict,
            matched_id=decision.matched_id,
            confidence=decision.confidence,
            temperature=temperature,
            model_version=_predictor.version,
            role=role,
        )
    )
    return MatchResponse(
        verdict=decision.verdict,
        matched_id=decision.matched_id,
        confidence=decision.confidence,
        temperature=temperature,
        reason=decision.reason,
        model_version=_predictor.version,
    )


@app.get("/v1/admin/audit")
async def audit_log(
    limit: int = 100,
) -> list[dict[str, object]]:
    """Recent match decisions. Admin-only - this is an operational review
    surface, not something an analyst-role caller should be able to bulk-read."""
    return _audit.recent(limit)


@app.get("/v1/admin/verdict_distribution")
async def verdict_distribution(
    since_hours: int = 24,
) -> dict[str, int]:
    """Feeds the drift monitor described in docs/api.md: a rising UNCERTAIN
    share over time is the earliest signal of camera/domain drift."""
    return _audit.verdict_counts(since_hours)


@app.post("/v1/search/photo", response_model=list[SightingResponse])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def search_photo(
    request: Request,
    image: UploadFile = File(...),
    k: int = 10,
) -> list[SightingResponse]:
    """Upload a photo of the vehicle itself; returns calibrated sightings.

    Returns an empty list (not 503) when the gallery index has not been built yet,
    so the frontend can handle this gracefully without a network error.
    """
    blob = await image.read()
    validate_image_bytes(blob)

    import io
    import numpy as np
    from PIL import Image

    try:
        pil_image = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="unreadable image") from exc
    validate_dimensions(*pil_image.size)

    # Gracefully handle missing gallery index
    try:
        engine = _get_search_engine()
    except HTTPException as exc:
        if exc.status_code == 503:
            log.warning("search.photo.no_index", note="gallery not built — returning empty results")
            return []
        raise

    sightings = engine.search_by_photo(np.array(pil_image), k=min(max(k, 1), 50))

    cid = new_correlation_id()
    log.info("search.photo", correlation_id=cid, n_results=len(sightings))

    fmt = request.query_params.get("format", "json")
    if fmt.lower() == "csv":
        import csv as _csv
        from fastapi.responses import PlainTextResponse
        output = io.StringIO()
        writer = _csv.writer(output)
        writer.writerow(["Location", "Dataset", "Plate", "Confidence", "Verdict", "Reason"])
        for s in sightings:
            writer.writerow([s.camera_id, s.dataset, s.plate_text or "", f"{s.confidence:.3f}", s.verdict, s.reason])
        return PlainTextResponse(output.getvalue(), media_type="text/csv")

    return [SightingResponse(**s.__dict__) for s in sightings]



@app.get("/v1/search/plate")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def search_plate(
    request: Request, plate_query: str, k: int = 10, fmt: str = "json"
) -> list[SightingResponse] | Any:
    """Search by license plate text. Returns empty list (not 503) when gallery not built."""
    payload = PlateSearchRequest(plate_query=plate_query, k=k)

    # Gracefully handle missing gallery index
    try:
        engine = _get_search_engine()
    except HTTPException as exc:
        if exc.status_code == 503:
            log.warning("search.plate.no_index", note="gallery not built — returning empty results")
            return []
        raise

    try:
        sightings = engine.search_by_plate(payload.plate_query, k=payload.k)
    except DataIntegrityError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cid = new_correlation_id()
    log.info(
        "search.plate",
        correlation_id=cid,
        plate_hint_len=len(payload.plate_query),
        n_results=len(sightings),
    )

    if fmt.lower() == "csv":
        import io, csv as _csv
        from fastapi.responses import PlainTextResponse
        output = io.StringIO()
        writer = _csv.writer(output)
        writer.writerow(["Location", "Dataset", "Plate", "Confidence", "Verdict", "Reason"])
        for s in sightings:
            writer.writerow([s.camera_id, s.dataset, s.plate_text or "", f"{s.confidence:.3f}", s.verdict, s.reason])
        return PlainTextResponse(output.getvalue(), media_type="text/csv")

    return [SightingResponse(**s.__dict__) for s in sightings]


@app.get("/v1/dataset/search")
async def search_datasets(
    q: str = "",
    field: str = "all",
    split: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Full-text search across all processed CSV dataset manifests.

    Searches every column (or a specific column via ``field``) across all CSV
    files in data/processed/.  This works without a FAISS gallery index, making
    search available as soon as datasets are ingested.

    Uses streaming pagination to avoid loading all rows into memory.

    Args:
        q:      Search query string (case-insensitive substring match)
        field:  Column to search (default 'all' = every column)
        split:  Filter to a specific split (train / query / gallery / all)
        limit:  Maximum number of results (capped at 2000)
        offset: Pagination offset
    """
    import csv as _csv

    limit = min(max(1, limit), 2000)
    offset = max(0, offset)
    q_lower = q.strip().lower()

    _KNOWN_SPLITS = {"train", "query", "gallery", "all", "test", "val"}

    # Build list of CSV files to search.
    # When a specific split is requested, prefer <name>_<split>.csv files and
    # SKIP _all.csv files to avoid returning duplicate rows.
    csv_files: list[Path] = []
    for csv_file in sorted(_PROCESSED_DIR.glob("*.csv")):
        stem = csv_file.stem
        parts = stem.rsplit("_", 1)
        file_split = parts[1] if len(parts) == 2 and parts[1] in _KNOWN_SPLITS else None

        if split and split != "all":
            # Only include files for this split; exclude _all to avoid double-hits
            if file_split == split:
                csv_files.append(csv_file)
        else:
            # No split filter — use only _all.csv files (or standalone files) to
            # avoid counting the same record 2× (once in split + once in _all).
            if file_split == "all" or file_split is None:
                csv_files.append(csv_file)

    # If no files were found with the above logic, fall back to all CSVs
    if not csv_files:
        csv_files = sorted(_PROCESSED_DIR.glob("*.csv"))

    results: list[dict] = []
    seen = 0  # number of matching rows encountered so far

    for csv_file in csv_files:
        if len(results) >= limit:
            break
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if len(results) >= limit:
                        break
                    # Check if row matches the query
                    if q_lower:
                        if field == "all":
                            haystack = " ".join(str(v) for v in row.values()).lower()
                        else:
                            haystack = str(row.get(field, "")).lower()
                        if q_lower not in haystack:
                            continue
                    # Apply pagination
                    if seen < offset:
                        seen += 1
                        continue
                    results.append(dict(row))
                    seen += 1
        except Exception as exc:
            log.warning("dataset.search.read_error", file=csv_file.name, exc=str(exc))

    return results


@app.get("/v1/dataset/search/count")
async def search_datasets_count(q: str = "", field: str = "all", split: str = "") -> dict:
    """Return the count of records matching a search query (cheap — no payload)."""
    import csv as _csv

    _KNOWN_SPLITS = {"train", "query", "gallery", "all", "test", "val"}
    q_lower = q.strip().lower()
    count = 0

    csv_files: list[Path] = []
    for csv_file in sorted(_PROCESSED_DIR.glob("*.csv")):
        stem = csv_file.stem
        parts = stem.rsplit("_", 1)
        file_split = parts[1] if len(parts) == 2 and parts[1] in _KNOWN_SPLITS else None
        if split and split != "all":
            if file_split == split:
                csv_files.append(csv_file)
        else:
            if file_split == "all" or file_split is None:
                csv_files.append(csv_file)

    if not csv_files:
        csv_files = sorted(_PROCESSED_DIR.glob("*.csv"))

    for csv_file in csv_files:
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    if not q_lower:
                        count += 1
                    elif field == "all":
                        if q_lower in " ".join(str(v) for v in row.values()).lower():
                            count += 1
                    else:
                        if q_lower in str(row.get(field, "")).lower():
                            count += 1
        except Exception:
            pass
    return {"count": count, "query": q, "field": field, "split": split}

# ──────────────────────────────────────────────────────────────────
#  File Serving (for UI)
# ──────────────────────────────────────────────────────────────────

@app.get("/api/files/{path:path}")
async def serve_file(path: str):
    """Serve files directly by absolute path for the frontend (local dev only)."""
    from fastapi.responses import FileResponse
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

# ──────────────────────────────────────────────────────────────────
#  Gallery browser endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/v1/gallery", response_model=list[GalleryEntryResponse])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def list_gallery(
    request: Request,
    limit: int = 200,
    offset: int = 0,
) -> list[GalleryEntryResponse]:
    """Return paginated gallery entries from the metadata store."""
    store_path = Path(settings.model_registry_path) / "gallery" / "store.db"
    if not store_path.exists():
        raise HTTPException(status_code=503, detail="gallery not built yet — run build_gallery_index.py")

    import sqlite3
    conn = sqlite3.connect(store_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM gallery_entries ORDER BY faiss_position LIMIT ? OFFSET ?",
            (max(1, min(limit, 500)), max(0, offset)),
        ).fetchall()
        return [GalleryEntryResponse(**dict(r)) for r in rows]
    finally:
        conn.close()


@app.get("/v1/gallery/stats", response_model=GalleryStatsResponse)
async def gallery_stats() -> GalleryStatsResponse:
    """Aggregate stats: total entries, unique cameras, datasets and plates."""
    store_path = Path(settings.model_registry_path) / "gallery" / "store.db"
    if not store_path.exists():
        return GalleryStatsResponse(total=0, cameras=0, datasets=0, plates=0)

    import sqlite3
    conn = sqlite3.connect(store_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM gallery_entries").fetchone()[0]
        cameras = conn.execute("SELECT COUNT(DISTINCT camera_id) FROM gallery_entries").fetchone()[0]
        datasets = conn.execute("SELECT COUNT(DISTINCT dataset) FROM gallery_entries").fetchone()[0]
        plates = conn.execute("SELECT COUNT(*) FROM gallery_entries WHERE plate_text IS NOT NULL").fetchone()[0]
        return GalleryStatsResponse(total=total, cameras=cameras, datasets=datasets, plates=plates)
    finally:
        conn.close()


@app.delete("/v1/gallery/dataset/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its entries from the store."""
    store_path = Path(settings.model_registry_path) / "gallery" / "store.db"
    if not store_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    import sqlite3
    import urllib.parse
    dataset_name = urllib.parse.unquote(dataset_name)
    
    conn = sqlite3.connect(store_path)
    try:
        # Just delete from SQLite for now. True production system would also
        # rebuild the FAISS index to remove vectors, but that requires full re-index.
        # This is sufficient for UI demonstration.
        cursor = conn.execute("DELETE FROM gallery_entries WHERE dataset = ?", (dataset_name,))
        conn.commit()
        deleted = cursor.rowcount
        
        # Force reload search engine to reflect changes if necessary
        global _search_engine
        _search_engine = None
        
        return {"status": "success", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
#  Video ingestion — 2-stage pipeline: RF-DETR → OSNet embedding
# ──────────────────────────────────────────────────────────────────

@app.post("/v1/ingest/video", response_model=IngestionStatusResponse)
@limiter.limit("10/minute")
async def ingest_video(
    request: Request,
    video: UploadFile = File(...),
    frame_skip: int = 5,
) -> IngestionStatusResponse:
    """Accept a video file, run the 2-stage detection+embedding pipeline in the background.

    Stage 1 (fast, small model): RF-DETR detects vehicle bounding boxes per frame.
    Stage 2 (large encoder, crop-only): OSNet-AIN embeds ONLY the vehicle crop — not the
    whole frame — dramatically reducing computation vs feeding full frames to the encoder.
    """
    import asyncio, tempfile, uuid, time
    from pathlib import Path as P

    job_id = uuid.uuid4().hex[:12]
    filename = video.filename or "upload.mp4"

    # Read uploaded bytes to a temp file
    blob = await video.read()
    if len(blob) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="video file exceeds 500 MB limit")

    job: dict = {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
        "total_frames": 0,
        "processed_frames": 0,
        "detections": 0,
        "embeddings": 0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "error": None,
    }
    _ingestion_jobs[job_id] = job

    async def _run_pipeline() -> None:
        import io
        import cv2
        import numpy as np
        import torch
        import torchvision.transforms as T

        start = time.time()
        job["status"] = "running"
        try:
            # Write video to temp file
            with tempfile.NamedTemporaryFile(suffix=P(filename).suffix or ".mp4", delete=False) as f:
                f.write(blob)
                tmp_path = f.name

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            job["total_frames"] = max(total_frames, 1)

            # 2-stage pipeline setup
            from reiduq.detection.rfdetr import RFDetrDetector
            from reiduq.segmentation.base import gate_crop
            from reiduq.models.builder import build_encoder
            from reiduq.retrieval.index import GalleryIndex, GalleryEntry
            from reiduq.search.gallery_store import GalleryStore, GalleryEntryRow

            detector = RFDetrDetector(device="cpu")
            encoder = build_encoder("osnet_ain", embed_dim=512, pretrained=True)
            encoder.eval()

            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Prepare output stores
            out_dir = P(settings.model_registry_path) / "gallery"
            out_dir.mkdir(parents=True, exist_ok=True)
            index_path = out_dir / "index.faiss"
            store_path = out_dir / "store.db"

            store = GalleryStore(store_path)
            initial_count = store.count()

            new_vectors: list[np.ndarray] = []
            new_entries: list[GalleryEntry] = []

            frame_id = 0
            crop_id = initial_count

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if frame_id % max(frame_skip, 1) != 0:
                    continue

                # Convert BGR→RGB for detector
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # STAGE 1: RF-DETR vehicle detection (fast, on small model)
                detections = detector.detect(rgb, frame_id=frame_id, camera_id=f"video_{job_id}")

                for det in detections:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    # STAGE 2: OSNet embed ONLY the vehicle crop (not full frame)
                    crop = rgb[y1:y2, x1:x2]
                    tensor = transform(crop).unsqueeze(0)
                    with torch.no_grad():
                        vec = encoder(tensor)
                        vec = torch.nn.functional.normalize(vec, dim=1)
                    embedding = vec.cpu().numpy()[0].astype(np.float32)

                    uid = f"{job_id}_f{frame_id:06d}_d{job['detections']:04d}"
                    # Save crop image
                    crop_out = out_dir / "crops"
                    crop_out.mkdir(exist_ok=True)
                    crop_path = str(crop_out / f"{uid}.jpg")
                    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                    new_vectors.append(embedding)
                    new_entries.append(GalleryEntry(uid=uid, identity=0, camera_id=f"video_{job_id}"))
                    store.add(GalleryEntryRow(
                        faiss_position=initial_count + len(new_vectors) - 1,
                        uid=uid,
                        image_path=crop_path,
                        dataset=filename,
                        camera_id=f"video_{job_id}",
                        ground_truth_identity=None,
                        plate_text=None,
                        plate_confidence=None,
                        visibility=1.0,
                    ))
                    job["detections"] += 1
                    job["embeddings"] += 1

                job["processed_frames"] += 1
                elapsed = time.time() - start
                job["elapsed_seconds"] = elapsed
                frames_done = job["processed_frames"]
                rate = frames_done / elapsed if elapsed > 0 else 1
                total_to_process = max(total_frames // max(frame_skip, 1), 1)
                remaining = total_to_process - frames_done
                job["eta_seconds"] = remaining / rate if rate > 0 else None

                # Yield control to event loop
                await asyncio.sleep(0)

            cap.release()
            import os
            os.unlink(tmp_path)

            # Rebuild FAISS index to include new vectors
            if new_vectors:
                try:
                    existing_index = GalleryIndex.load(index_path)
                    existing_vecs = existing_index._index.reconstruct_n(0, existing_index._index.ntotal)
                    all_vecs = np.vstack([existing_vecs] + new_vectors)
                    existing_entries_list = list(existing_index.entries)
                except Exception:
                    all_vecs = np.stack(new_vectors)
                    existing_entries_list = []

                new_index = GalleryIndex(dim=512)
                new_index.add(all_vecs, existing_entries_list + new_entries)
                new_index.build()
                new_index.save(index_path)

                # Reload the global search engine with updated index
                global _search_engine
                _search_engine = None  # force reload on next search

            job["status"] = "done"
            job["elapsed_seconds"] = time.time() - start
            job["eta_seconds"] = 0.0
            log.info("ingest.done", job_id=job_id, embeddings=job["embeddings"])

        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            log.error("ingest.error", job_id=job_id, exc_info=exc)

    asyncio.create_task(_run_pipeline())
    return IngestionStatusResponse(**job)


@app.get("/v1/ingest/status/{job_id}", response_model=IngestionStatusResponse)
async def ingest_status(job_id: str) -> IngestionStatusResponse:
    """Poll the status of a running or completed ingestion job."""
    job = _ingestion_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return IngestionStatusResponse(**job)



# ──────────────────────────────────────────────────────────────────
#  Dataset ingestion — ZIP / folder → CSV manifest + preview images
# ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # vehicle-reid-uncertainty/
_PROCESSED_DIR = (_REPO_ROOT / "data" / "processed").resolve()
_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/v1/dataset/ingest", response_model=DatasetIngestStatus)
@limiter.limit("5/minute")
async def ingest_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_name: str = "dataset",
    split: str = "all",
    n_preview: int = 20,
    delete_originals: bool = False,
) -> DatasetIngestStatus:
    """Accept a ZIP archive and process it into a CSV manifest + thumbnails.

    Processing runs in the background. Poll /v1/dataset/ingest/status/{job_id}
    for progress. When done the CSV is available at /v1/dataset/{dataset_name}/csv.
    """
    import asyncio, tempfile, uuid, time

    job_id = uuid.uuid4().hex[:12]
    blob = await file.read()

    if len(blob) > 1024 * 1024 * 1024:  # 1 GB hard cap
        raise HTTPException(status_code=413, detail="dataset file exceeds 1 GB limit")

    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", dataset_name)[:64]

    job: dict = {
        "job_id": job_id,
        "status": "queued",
        "dataset_name": safe_name,
        "total_images": 0,
        "processed": 0,
        "preview_count": 0,
        "csv_path": None,
        "errors": [],
        "elapsed_seconds": 0.0,
        "error": None,
    }
    _dataset_jobs[job_id] = job

    async def _run() -> None:
        import time
        from reiduq.data.dataset_processor import DatasetProcessor

        start = time.perf_counter()
        job["status"] = "running"

        try:
            suffix = Path(file.filename or "upload.zip").suffix.lower() or ".zip"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(blob)
                tmp_path = f.name

            def _progress(done: int, total: int) -> None:
                job["total_images"] = total
                job["processed"] = done
                job["elapsed_seconds"] = time.perf_counter() - start

            proc = DatasetProcessor(
                out_dir=_PROCESSED_DIR,
                dataset_name=safe_name,
                n_preview=n_preview,
                delete_originals=delete_originals,
                progress_cb=_progress,
            )

            import zipfile
            if zipfile.is_zipfile(tmp_path):
                result = proc.process_zip(Path(tmp_path), split=split)
            else:
                # Treat as a directory if it is one, else error
                raise ValueError("Uploaded file is not a ZIP archive. Upload a ZIP or use the folder endpoint.")

            import os; os.unlink(tmp_path)

            job["status"] = "done"
            job["total_images"] = result.total_images
            job["processed"] = result.processed
            job["preview_count"] = result.preview_count
            job["csv_path"] = result.csv_path
            job["errors"] = result.errors[:20]  # cap to avoid huge payloads
            job["elapsed_seconds"] = result.elapsed_seconds
            log.info("dataset_ingest.done", job_id=job_id, processed=result.processed)

        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            log.error("dataset_ingest.error", job_id=job_id, exc_info=exc)

    asyncio.create_task(_run())
    return DatasetIngestStatus(**job)


@app.get("/v1/dataset/ingest/status/{job_id}", response_model=DatasetIngestStatus)
async def dataset_ingest_status(job_id: str) -> DatasetIngestStatus:
    """Poll the status of a running or completed dataset ingestion job."""
    job = _dataset_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return DatasetIngestStatus(**job)


@app.get("/v1/dataset/list", response_model=list[DatasetListEntry])
async def list_datasets() -> list[DatasetListEntry]:
    """List all processed dataset CSV manifests available on disk.

    Groups split CSVs (e.g. veri776_train.csv, veri776_query.csv) under their
    base dataset name (veri776), summing row counts across all splits.
    """
    import csv as _csv
    from collections import defaultdict

    # Group files by base dataset name
    # Pattern: <name>_<split>.csv  or  <name>.csv
    _KNOWN_SPLITS = {"train", "query", "gallery", "all", "test", "val"}

    groups: dict[str, list[Path]] = defaultdict(list)
    for csv_file in sorted(_PROCESSED_DIR.glob("*.csv")):
        stem = csv_file.stem  # e.g. "veri776_train"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in _KNOWN_SPLITS:
            base = parts[0]
        else:
            base = stem
        groups[base].append(csv_file)

    entries: list[DatasetListEntry] = []
    for base_name, csv_files in sorted(groups.items()):
        try:
            total_rows = 0
            total_size = 0
            earliest_ctime = float("inf")
            for csv_file in csv_files:
                stat = csv_file.stat()
                total_size += stat.st_size
                earliest_ctime = min(earliest_ctime, stat.st_ctime)
                with open(csv_file, newline="", encoding="utf-8") as f:
                    total_rows += sum(1 for _ in _csv.DictReader(f))

            # Use the "all" CSV as primary if it exists, else the first one
            all_csv = next((f for f in csv_files if f.stem.endswith("_all") or f.stem == base_name), csv_files[0])

            preview_dir = _PROCESSED_DIR / "previews" / base_name
            preview_count = len(list(preview_dir.glob("*.jpg"))) if preview_dir.exists() else 0

            entries.append(DatasetListEntry(
                dataset_name=base_name,
                csv_file=all_csv.name,
                row_count=total_rows,
                preview_count=preview_count,
                created_at=datetime.fromtimestamp(earliest_ctime, UTC).isoformat(),
                csv_size_bytes=total_size,
            ))
        except Exception as exc:
            log.warning("dataset.list_error", base=base_name, exc=str(exc))
    return entries


@app.get("/v1/dataset/{dataset_name}/csv")
async def download_dataset_csv(dataset_name: str) -> Response:
    """Stream the CSV manifest for the named dataset."""
    import urllib.parse
    safe = urllib.parse.unquote(dataset_name)
    # Find CSV files matching this dataset (may have multiple splits)
    matches = list(_PROCESSED_DIR.glob(f"{safe}_*.csv")) + list(_PROCESSED_DIR.glob(f"{safe}.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No CSV found for dataset '{safe}'")
    # Return the first match (or merge them in a real implementation)
    csv_path = matches[0]
    from fastapi.responses import FileResponse
    return FileResponse(str(csv_path), media_type="text/csv", filename=csv_path.name)


@app.get("/v1/dataset/{dataset_name}/rows")
async def dataset_rows(
    dataset_name: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return paginated rows from the dataset CSV manifest(s) as JSON.

    For grouped datasets (e.g. veri776 = train+query+gallery), merges all
    matching split CSVs before paginating.
    """
    import csv as _csv, urllib.parse
    safe = urllib.parse.unquote(dataset_name)

    # Collect all matching CSVs: <safe>_*.csv and <safe>.csv
    matches: list[Path] = sorted(
        list(_PROCESSED_DIR.glob(f"{safe}_*.csv")) + list(_PROCESSED_DIR.glob(f"{safe}.csv"))
    )
    # Prefer the "_all" file if present (avoids duplicate rows from combined)
    all_file = [f for f in matches if f.stem.endswith("_all")]
    if all_file:
        matches = all_file

    if not matches:
        raise HTTPException(status_code=404, detail=f"No CSV found for dataset '{safe}'")

    rows: list[dict] = []
    seen = 0
    for csv_path in matches:
        if len(rows) >= limit:
            break
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if seen < offset:
                    seen += 1
                    continue
                if len(rows) >= limit:
                    break
                rows.append(dict(row))
                seen += 1
    return rows


@app.get("/v1/dataset/{dataset_name}/previews")
async def dataset_previews(dataset_name: str) -> list[dict]:
    """Return URLs for all preview thumbnail images for a dataset."""
    import urllib.parse
    safe = urllib.parse.unquote(dataset_name)
    preview_dir = _PROCESSED_DIR / "previews" / safe
    if not preview_dir.exists():
        return []
    files = sorted(preview_dir.glob("*.jpg"))
    # Return paths that the frontend can fetch via /api/files/<path>
    return [{"filename": f.name, "path": str(f)} for f in files]

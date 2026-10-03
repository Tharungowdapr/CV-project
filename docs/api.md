# API

Base URL: `http://localhost:8000`. All routes except `/health` require a credential - either
`Authorization: Bearer <key>` (server-to-server callers) or a session cookie obtained from
`/v1/auth/login` (the browser console). Keys and roles come from the `API_KEYS` environment
variable.

## `POST /v1/auth/login`
Public, but rate-limited to 10/minute (tighter than the general limit - this is the
credential-guessing surface). Exchanges an API key for an httpOnly, `SameSite=Strict` session
cookie valid for 8 hours. The response never distinguishes a wrong key from an unknown one.

```json
{ "api_key": "devkey" }
```
```json
{ "role": "analyst", "expires_in_seconds": 28800 }
```

## `POST /v1/auth/logout`
Requires a valid credential. Clears the session cookie.

## `GET /v1/admin/audit?limit=100`
Role: `admin` only. Returns the most recent match decisions from the audit log (verdict,
confidence, ids - never images or raw similarity vectors). `limit` is clamped to 1000.

## `GET /v1/admin/verdict_distribution?since_hours=24`
Role: `admin` only. Verdict counts over the window, grouped. A rising `UNCERTAIN` share here is
the earliest signal of camera or domain drift - see docs/reproducibility.md and the operations
guide's monitoring section.

## `POST /v1/search/photo`
Role: `analyst` or `admin`. Multipart upload (`image`, JPEG/PNG, same size/dimension limits as
elsewhere) of a photo of the vehicle itself. Embeds it, searches the built gallery index, and
returns calibrated sightings - never a bare similarity list. Returns **503** if no gallery index
has been built yet (`scripts/build_gallery_index.py` has not been run).

```json
[
  {
    "image_path": "data/raw/VeRi/image_train/0002_c002_00030600_0.jpg",
    "camera_id": "c002",
    "dataset": "veri776",
    "similarity": 0.91,
    "confidence": 0.74,
    "verdict": "UNCERTAIN",
    "reason": "confidence 0.740 in the ambiguous band [0.200, 0.900) - route to human review",
    "plate_text": null,
    "plate_confidence": null
  }
]
```

## `GET /v1/search/plate?plate_query=KA05&k=10`
Role: `analyst` or `admin`. Fuzzy (case-insensitive substring) match against OCR-read plate text
in the gallery index. On a hit, the matched image becomes a Re-ID query anchor and the response
is every appearance-matched sighting of that vehicle - **including ones where the plate was
never legible** - not a bare list of plate-text matches. Returns **404** if nothing in the index
has a plate reading that matches.

Read this before trusting a result: the source datasets carry no ground-truth plate labels and
were not built with plate legibility in mind, so `plate_text` is a best-effort OCR read with its
own `plate_confidence`, propagated onto every result in the response - never treat it as a
verified identifier.

## `GET /health`
Public. Used by the container healthcheck and the console status indicator.

```json
{ "status": "ok", "version": "0.1.0", "model_loaded": true, "calibrator": "ConditionalTemperature" }
```

`model_loaded: false` means the service started without a checkpoint — it is up but not
deployed, and the distinction matters when debugging.

## `POST /v1/match`
Role: `analyst` or `admin`. Rate limited per client.

```json
{
  "query_id": "c012_q0001",
  "similarities": [0.94, 0.88, 0.81],
  "candidate_ids": ["g0001", "g0002", "g0003"],
  "visibility": 0.85,
  "part_visibility": [1, 0.4, 1, 1, 1, 0.9],
  "camera_stats": [0.41, 0.39, 0.37, 0.18, 0.17, 0.16, 0.22, 0.31, 1.04],
  "n_frames": 12
}
```

Response:

```json
{
  "verdict": "UNCERTAIN",
  "matched_id": "g0001",
  "confidence": 0.612,
  "temperature": 2.84,
  "reason": "confidence 0.612 in the ambiguous band [0.200, 0.900) - route to human review",
  "model_version": "calibrator"
}
```

`similarities` must be descending and aligned with `candidate_ids`. `query_id` matches
`^[A-Za-z0-9_.:-]+$`; anything else is a 422. Unknown fields are rejected rather than ignored.

## Errors

| Status | Meaning |
|---|---|
| 401 | Missing or invalid bearer token |
| 403 | Authenticated but the role is insufficient |
| 413 | Payload or image exceeds a limit |
| 415 | Unsupported media type |
| 422 | Schema validation failed |
| 429 | Rate limit exceeded |
| 500 | Internal error — returns a correlation id only; the detail is in the logs under that id |

Error bodies never contain stack traces, internal paths, or model internals.

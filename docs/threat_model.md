# Threat model

Scope: the training pipeline, the inference API, and the review console. Out of scope: the
network perimeter and the identity provider, which are the deployer's responsibility.

## Assets and controls

| Asset | Threat | Control |
|---|---|---|
| Model weights | Exfiltration, tampering | Checksummed artifacts; signed release images; weights never in git (DVC/LFS); the service mounts `models/` read-only |
| Checkpoints | **Arbitrary code execution** — `torch.load` unpickles, and unpickling executes | `torch.load(..., weights_only=True)` everywhere; `.pkl` from untrusted paths is rejected outright |
| Uploaded images | Decompression bomb, malicious EXIF, path traversal via filename | Size cap, pixel-count cap, magic-byte check, re-encode before use, filename never trusted |
| Dataset paths from config | Arbitrary file read | `safe_path()` resolves and asserts containment under `DATA_ROOT` |
| Inference API | DoS, scraping, unauthorised use | Per-client rate limit, request size limits, bearer auth on every route except `/health`, role separation |
| Dependencies | Supply-chain compromise | Exact pins, `pip-audit`, Dependabot, trivy, SBOM per release, cosign signatures |
| Logs | PII / plate leakage | Ids and hashes only; raw crops never logged; user data bound as key-value pairs, never interpolated into messages |
| Audit database | Same leakage risk as logs, plus a persistent store to exfiltrate | Same fields as logs (ids, verdict, confidence) - never images or raw similarity vectors; SQLite file is not world-readable by default |
| Session cookie | Forged role, replay after expiry, XSS exfiltration | HMAC-signed with `SECRET_KEY` (a forged role fails signature check); embedded expiry checked server-side; `httponly` (JS cannot read it even via XSS) and `SameSite=Strict` (never sent cross-site) |
| Rate limiter | Per-replica-only counting silently multiplies the effective limit | `REDIS_URL` moves the counter to a shared backend; unset in single-replica deployments only |
| Config files | Injection through YAML | `yaml.safe_load` only; Pydantic `extra="forbid"` rejects unknown keys |
| The system itself | Surveillance misuse | Model card with intended-use restrictions; abstention on by default; documented human-review requirement for UNCERTAIN |

## Trust boundaries

1. **Internet → API.** Everything crossing it is validated by Pydantic. No field reaches the
   filesystem or a model without passing a schema.
2. **Config file → pipeline.** Configs are trusted to the extent that whoever can write one can
   already run code — but paths are still contained, because a config is the most likely place
   for a copy-pasted mistake.
3. **Checkpoint → process.** Treated as untrusted input. This is the boundary most ML codebases
   ignore.

## Explicit non-goals

- Model-extraction resistance. An authorised caller can query the API.
- Adversarial-patch robustness against a physical attacker.
- Privacy guarantees for the vehicles in public research datasets beyond their licence terms.

Naming these is deliberate: an unstated non-goal reads as an oversight.

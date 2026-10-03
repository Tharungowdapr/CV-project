"""Runtime settings read from the environment. No secret is ever hardcoded."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"
    log_format: str = "json"

    secret_key: str = Field(default="change-me", min_length=8)
    api_keys: str = ""  # "key:role,key:role" - parsed, never logged
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    rate_limit_per_minute: int = 600
    max_upload_bytes: int = 500 * 1024 * 1024  # 500 MB to allow video uploads
    max_image_pixels: int = 4096 * 4096

    # In-memory rate limiting (the default) only counts requests seen by the
    # single process it runs in. Behind a load balancer with more than one
    # API replica, each replica enforces the limit independently, so the
    # *effective* limit becomes rate_limit_per_minute * n_replicas without
    # anyone deciding that on purpose. Setting redis_url points slowapi at a
    # shared backend so the limit means what it says regardless of replica
    # count. Unset in single-instance/dev deployments; the service still
    # works, just with that one caveat.
    redis_url: str | None = None

    audit_db_path: str = "./outputs/audit.db"  # SQLite log of match decisions; see serving/audit.py

    data_root: str = "./data"
    artifact_root: str = "./outputs"
    model_registry_path: str = "./models"

    device: str = "cuda"
    num_workers: int = 8
    seed: int = 42

    @property
    def api_key_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            if ":" in pair:
                key, role = pair.split(":", 1)
                out[key.strip()] = role.strip()
        return out

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()

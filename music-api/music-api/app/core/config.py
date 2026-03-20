from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────
    app_name: str = "Music Recommendation API"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Security ──────────────────────────────────────────────────────
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("CHANGE-ME-IN-PRODUCTION-USE-256BIT-SECRET"),
        description="JWT signing key — must be a strong random secret in production",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1, le=10080)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=90)

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: SecretStr = Field(
        default=SecretStr("redis://localhost:6379/0"),
    )
    cache_ttl_seconds: int = Field(default=300, ge=10, le=86400)
    recommendations_cache_ttl: int = Field(default=600, ge=10, le=3600)

    # ── Graph ─────────────────────────────────────────────────────────
    graph_max_nodes: int = Field(default=1_000_000, ge=100)
    graph_max_recommendations: int = Field(default=50, ge=1, le=200)
    pagerank_alpha: float = Field(default=0.85, ge=0.1, le=0.99)
    similarity_min_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    embedding_dimensions: int = Field(default=128, ge=16, le=512)

    # ── Rate limiting ─────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_auth_per_minute: int = Field(default=10, ge=1, le=100)

    # ── CORS ──────────────────────────────────────────────────────────
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"]
    )

    @field_validator("environment")
    @classmethod
    def warn_insecure_production(cls, v: str) -> str:
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def redis_url_safe(self) -> str:
        """URL sem credenciais para logging."""
        url = self.redis_url.get_secret_value()
        if "@" in url:
            scheme, rest = url.split("//", 1)
            rest = rest.split("@", 1)[1]
            return f"{scheme}//**:***@{rest}"
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.middleware.error_handler import (
    RequestLoggingMiddleware,
    app_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.api.routes import auth, health, recommendations, tracks
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import configure_logging, get_logger
from app.graph.engine import MusicGraph
from app.repositories.base import InMemoryTrackRepository, InMemoryUserRepository
from app.services.cache import InMemoryCacheService, RedisCacheService

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── STARTUP ──────────────────────────────────────────────────────
    configure_logging()
    logger.info(
        "starting_up",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # Graph
    app.state.music_graph = MusicGraph()

    # Repositories
    app.state.user_repo = InMemoryUserRepository()
    app.state.track_repo = InMemoryTrackRepository()

    # Cache — tenta Redis, cai para in-memory se falhar
    try:
        cache = RedisCacheService(settings.redis_url.get_secret_value())
        if await cache.ping():
            app.state.cache = cache
            logger.info("cache_backend", backend="redis", url=settings.redis_url_safe)
        else:
            raise ConnectionError("Redis ping falhou")
    except Exception as exc:
        logger.warning("redis_unavailable_using_fallback", error=str(exc))
        app.state.cache = InMemoryCacheService()

    logger.info("startup_complete", graph_stats=app.state.music_graph.stats)

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────
    logger.info("shutting_down")
    if isinstance(app.state.cache, RedisCacheService):
        await app.state.cache.close()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API de recomendação musical baseada em análise de grafos. "
            "Utiliza PageRank personalizado, collaborative filtering (SVD) "
            "e similaridade de conteúdo por audio features."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # Logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers — versionados
    v1_prefix = "/api/v1"
    app.include_router(health.router)
    app.include_router(auth.router, prefix=v1_prefix)
    app.include_router(recommendations.router, prefix=v1_prefix)
    app.include_router(tracks.router, prefix=v1_prefix)

    return app


app = create_app()

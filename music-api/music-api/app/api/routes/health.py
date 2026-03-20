from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.dependencies import Cache, Graph
from app.core.config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime
    dependencies: dict[str, str]
    graph_stats: dict[str, int]


@router.get("/health", response_model=HealthStatus)
async def health_check(
    graph: Graph,
    cache: Cache,
) -> HealthStatus:
    cache_ok = await cache.ping()

    return HealthStatus(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
        dependencies={
            "cache": "ok" if cache_ok else "degraded",
            "graph": "ok",
        },
        graph_stats=graph.stats,
    )


@router.get("/health/ready")
async def readiness() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}

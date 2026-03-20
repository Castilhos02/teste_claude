from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.domain.models import (
    Genre,
    InteractionType,
    Track,
    TrackCreate,
    User,
    UserCreate,
)
from app.graph.engine import MusicGraph
from app.main import create_app
from app.repositories.base import InMemoryTrackRepository, InMemoryUserRepository
from app.services.cache import InMemoryCacheService


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def app():
    _app = create_app()
    async with _app.router.lifespan_context(_app):
        yield _app


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest_asyncio.fixture
async def music_graph() -> MusicGraph:
    return MusicGraph()


@pytest_asyncio.fixture
async def user_repo() -> InMemoryUserRepository:
    return InMemoryUserRepository()


@pytest_asyncio.fixture
async def track_repo() -> InMemoryTrackRepository:
    return InMemoryTrackRepository()


@pytest_asyncio.fixture
async def cache() -> InMemoryCacheService:
    return InMemoryCacheService()


# ── Domain factories ──────────────────────────────────────────────────────────

def make_user_create(
    username: str | None = None,
    email: str | None = None,
    password: str = "Senha123",
) -> UserCreate:
    suffix = str(uuid4())[:8]
    return UserCreate(
        username=username or f"user_{suffix}",
        email=email or f"user_{suffix}@test.com",
        display_name="Test User",
        password=password,
        confirm_password=password,
    )


def make_track_create(
    title: str | None = None,
    genres: list[Genre] | None = None,
    popularity: float = 0.7,
) -> TrackCreate:
    return TrackCreate(
        title=title or f"Track {uuid4().hex[:6]}",
        artist_id=uuid4(),
        duration_ms=210_000,
        genres=genres or [Genre.POP],
        danceability=0.7,
        energy=0.8,
        valence=0.6,
        acousticness=0.2,
        instrumentalness=0.1,
        tempo_bpm=128.0,
        loudness_db=-8.0,
        popularity=popularity,
    )


@pytest_asyncio.fixture
async def registered_user(client: AsyncClient) -> dict:
    payload = make_user_create()
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "username": payload.username,
            "email": payload.email,
            "display_name": payload.display_name,
            "password": "Senha123",
            "confirm_password": "Senha123",
        },
    )
    assert resp.status_code == 201
    return resp.json()


@pytest_asyncio.fixture
def auth_headers(registered_user: dict) -> dict[str, str]:
    token = registered_user["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}

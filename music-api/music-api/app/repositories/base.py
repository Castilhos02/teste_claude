"""
Repositories — camada de persistência com interface abstrata.
Substitua InMemory* por implementações PostgreSQL/MongoDB sem mudar
a camada de serviço.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from app.domain.models import Track, TrackCreate, User, UserCreate
from app.core.security import hash_password

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> T | None: ...

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> list[T]: ...

    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool: ...


# ── User Repository ───────────────────────────────────────────────────────────

class AbstractUserRepository(BaseRepository[User]):
    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None: ...

    @abstractmethod
    async def create(self, payload: UserCreate) -> User: ...

    @abstractmethod
    async def update_embedding(self, user_id: UUID, embedding: list[float]) -> None: ...


class InMemoryUserRepository(AbstractUserRepository):
    def __init__(self) -> None:
        self._store: dict[UUID, User] = {}
        self._email_index: dict[str, UUID] = {}
        self._username_index: dict[str, UUID] = {}

    async def get_by_id(self, entity_id: UUID) -> User | None:
        return self._store.get(entity_id)

    async def get_by_email(self, email: str) -> User | None:
        uid = self._email_index.get(email.lower())
        return self._store.get(uid) if uid else None

    async def get_by_username(self, username: str) -> User | None:
        uid = self._username_index.get(username.lower())
        return self._store.get(uid) if uid else None

    async def create(self, payload: UserCreate) -> User:
        user = User(
            username=payload.username,
            email=payload.email,
            display_name=payload.display_name,
            hashed_password=hash_password(payload.password.get_secret_value()),
        )
        self._store[user.id] = user
        self._email_index[user.email.lower()] = user.id
        self._username_index[user.username.lower()] = user.id
        return user

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[User]:
        all_users = list(self._store.values())
        return all_users[offset : offset + limit]

    async def delete(self, entity_id: UUID) -> bool:
        user = self._store.pop(entity_id, None)
        if user:
            self._email_index.pop(user.email.lower(), None)
            self._username_index.pop(user.username.lower(), None)
            return True
        return False

    async def update_embedding(self, user_id: UUID, embedding: list[float]) -> None:
        if user_id in self._store:
            self._store[user_id].embedding = embedding


# ── Track Repository ──────────────────────────────────────────────────────────

class AbstractTrackRepository(BaseRepository[Track]):
    @abstractmethod
    async def create(self, payload: TrackCreate) -> Track: ...

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[Track]: ...

    @abstractmethod
    async def get_by_artist(self, artist_id: UUID, limit: int = 50) -> list[Track]: ...

    @abstractmethod
    async def update_embedding(self, track_id: UUID, embedding: list[float]) -> None: ...


class InMemoryTrackRepository(AbstractTrackRepository):
    def __init__(self) -> None:
        self._store: dict[UUID, Track] = {}

    async def get_by_id(self, entity_id: UUID) -> Track | None:
        return self._store.get(entity_id)

    async def create(self, payload: TrackCreate) -> Track:
        track = Track(**payload.model_dump())
        self._store[track.id] = track
        return track

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Track]:
        all_tracks = list(self._store.values())
        return all_tracks[offset : offset + limit]

    async def search(self, query: str, limit: int = 20) -> list[Track]:
        q = query.lower()
        return [
            t for t in self._store.values()
            if q in t.title.lower()
        ][:limit]

    async def get_by_artist(self, artist_id: UUID, limit: int = 50) -> list[Track]:
        return [
            t for t in self._store.values()
            if t.artist_id == artist_id
        ][:limit]

    async def delete(self, entity_id: UUID) -> bool:
        return self._store.pop(entity_id, None) is not None

    async def update_embedding(self, track_id: UUID, embedding: list[float]) -> None:
        if track_id in self._store:
            self._store[track_id].embedding = embedding

    async def get_by_ids(self, ids: list[UUID]) -> list[Track]:
        return [self._store[i] for i in ids if i in self._store]

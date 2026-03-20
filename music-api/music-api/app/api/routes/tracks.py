from uuid import UUID

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, Graph, TrackRepo
from app.core.exceptions import NotFoundException
from app.domain.models import Track, TrackCreate

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.post("", response_model=Track, status_code=201)
async def create_track(
    payload: TrackCreate,
    track_repo: TrackRepo,
    graph: Graph,
    _: CurrentUser,
) -> Track:
    track = await track_repo.create(payload)
    await graph.add_track(track)
    return track


@router.get("", response_model=list[Track])
async def list_tracks(
    track_repo: TrackRepo,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: CurrentUser = None,
) -> list[Track]:
    return await track_repo.list_all(limit=limit, offset=offset)


@router.get("/search", response_model=list[Track])
async def search_tracks(
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=50),
    track_repo: TrackRepo = None,
    _: CurrentUser = None,
) -> list[Track]:
    return await track_repo.search(q, limit=limit)


@router.get("/{track_id}", response_model=Track)
async def get_track(
    track_id: UUID,
    track_repo: TrackRepo,
    _: CurrentUser = None,
) -> Track:
    track = await track_repo.get_by_id(track_id)
    if not track:
        raise NotFoundException(f"Track {track_id} não encontrada")
    return track

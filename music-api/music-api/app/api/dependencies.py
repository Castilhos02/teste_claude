from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.core.security import TokenPayload, decode_token
from app.domain.models import User
from app.graph.engine import MusicGraph
from app.repositories.base import AbstractTrackRepository, AbstractUserRepository
from app.services.cache import CacheService
from app.services.recommendation import RecommendationService


_bearer = HTTPBearer(auto_error=False)


def get_music_graph(request: Request) -> MusicGraph:
    return request.app.state.music_graph


def get_user_repo(request: Request) -> AbstractUserRepository:
    return request.app.state.user_repo


def get_track_repo(request: Request) -> AbstractTrackRepository:
    return request.app.state.track_repo


def get_cache(request: Request) -> CacheService:
    return request.app.state.cache


def get_recommendation_service(
    graph: Annotated[MusicGraph, Depends(get_music_graph)],
    track_repo: Annotated[AbstractTrackRepository, Depends(get_track_repo)],
    user_repo: Annotated[AbstractUserRepository, Depends(get_user_repo)],
    cache: Annotated[CacheService, Depends(get_cache)],
) -> RecommendationService:
    return RecommendationService(graph, track_repo, user_repo, cache)


async def get_token_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenPayload:
    if not credentials:
        raise UnauthorizedException("Token de autenticação ausente")
    return decode_token(credentials.credentials, expected_type="access")


async def get_current_user(
    payload: Annotated[TokenPayload, Depends(get_token_payload)],
    user_repo: Annotated[AbstractUserRepository, Depends(get_user_repo)],
) -> User:
    user = await user_repo.get_by_id(UUID(payload.sub))
    if not user:
        raise UnauthorizedException("Usuário não encontrado")
    if not user.is_active:
        raise ForbiddenException("Conta desativada")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
RecommendationSvc = Annotated[RecommendationService, Depends(get_recommendation_service)]
UserRepo = Annotated[AbstractUserRepository, Depends(get_user_repo)]
TrackRepo = Annotated[AbstractTrackRepository, Depends(get_track_repo)]
Graph = Annotated[MusicGraph, Depends(get_music_graph)]
Cache = Annotated[CacheService, Depends(get_cache)]

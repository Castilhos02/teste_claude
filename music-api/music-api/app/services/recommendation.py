from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import UUID

from app.core.config import get_settings
from app.core.exceptions import NotFoundException, RecommendationException
from app.core.logging import get_logger
from app.domain.models import (
    Interaction,
    RecommendationReason,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedTrack,
    Track,
)
from app.graph.algorithms import GraphAlgorithms, ScoredNode

if TYPE_CHECKING:
    from app.graph.engine import MusicGraph
    from app.repositories.base import (
        AbstractTrackRepository,
        AbstractUserRepository,
    )
    from app.services.cache import CacheService

logger = get_logger(__name__)
settings = get_settings()


class RecommendationService:
    def __init__(
        self,
        music_graph: MusicGraph,
        track_repo: AbstractTrackRepository,
        user_repo: AbstractUserRepository,
        cache: CacheService,
    ) -> None:
        self._graph = music_graph
        self._tracks = track_repo
        self._users = user_repo
        self._cache = cache
        self._algo = GraphAlgorithms(music_graph)

    async def get_recommendations(
        self,
        user_id: UUID,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        user = await self._users.get_by_id(user_id)
        if not user:
            raise NotFoundException(f"Usuário {user_id} não encontrado")

        cache_key = self._build_cache_key(user_id, request)
        cached = await self._cache.get(cache_key)
        if cached:
            response = RecommendationResponse(**json.loads(cached))
            response.cached = True
            logger.info("recommendations_cache_hit", user_id=str(user_id))
            return response

        logger.info(
            "computing_recommendations",
            user_id=str(user_id),
            strategy=request.strategy,
            limit=request.limit,
        )

        scored_nodes = await self._run_algorithm(user_id, request)

        if not scored_nodes:
            logger.warning("no_recommendations_found", user_id=str(user_id))
            return RecommendationResponse(
                user_id=user_id,
                recommendations=[],
                algorithm_used=request.strategy,
            )

        recommendations = await self._hydrate_recommendations(
            scored_nodes, request, user_id
        )

        response = RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            algorithm_used=request.strategy,
            cached=False,
        )

        await self._cache.set(
            cache_key,
            response.model_dump_json(),
            ttl=settings.recommendations_cache_ttl,
        )

        return response

    async def _run_algorithm(
        self,
        user_id: UUID,
        request: RecommendationRequest,
    ) -> list[ScoredNode]:
        fetch_n = request.limit * 3

        match request.strategy:
            case RecommendationReason.GRAPH_PAGERANK:
                return self._algo.personalized_pagerank(user_id, fetch_n)
            case RecommendationReason.CONTENT_BASED:
                liked = [
                    UUID(tn.split(":")[1])
                    for tn, w in self._graph.get_user_tracks(user_id, min_weight=1.0)
                ][:20]
                if not liked:
                    raise RecommendationException(
                        "Usuário sem histórico suficiente para recomendação por conteúdo"
                    )
                return self._algo.content_based_similarity(liked, top_n=fetch_n)
            case RecommendationReason.COLLABORATIVE_FILTERING:
                return self._algo.collaborative_filtering(user_id, fetch_n)
            case RecommendationReason.SIMILAR_USERS:
                return self._algo.embedding_similarity(user_id, fetch_n)
            case _:
                return self._algo.hybrid_recommendations(user_id, fetch_n)

    async def _hydrate_recommendations(
        self,
        scored_nodes: list[ScoredNode],
        request: RecommendationRequest,
        user_id: UUID,
    ) -> list[RecommendedTrack]:
        excluded = {str(eid) for eid in request.exclude_track_ids}
        genre_filter = {g.value for g in request.genres} if request.genres else None

        results: list[RecommendedTrack] = []
        for node in scored_nodes:
            if len(results) >= request.limit:
                break
            if node.entity_id in excluded:
                continue

            track = await self._tracks.get_by_id(UUID(node.entity_id))
            if not track:
                continue
            if track.popularity < request.min_popularity:
                continue
            if genre_filter and not any(g.value in genre_filter for g in track.genres):
                continue

            results.append(
                RecommendedTrack(
                    track=track,
                    score=node.score,
                    reason=request.strategy,
                    explanation=self._build_explanation(request.strategy, track, node.score),
                )
            )

        return results

    def _build_explanation(
        self,
        strategy: RecommendationReason,
        track: Track,
        score: float,
    ) -> str:
        match strategy:
            case RecommendationReason.GRAPH_PAGERANK:
                return (
                    f"'{track.title}' é altamente conectada na rede de músicas "
                    f"que você ouve (score: {score:.3f})"
                )
            case RecommendationReason.CONTENT_BASED:
                return (
                    f"'{track.title}' tem características sonoras "
                    f"similares às suas músicas favoritas (similaridade: {score:.2%})"
                )
            case RecommendationReason.COLLABORATIVE_FILTERING:
                return (
                    f"Usuários com perfil parecido com o seu curtiram '{track.title}'"
                )
            case _:
                return (
                    f"'{track.title}' foi recomendada pela análise combinada "
                    f"do seu perfil musical (score: {score:.3f})"
                )

    async def record_interaction(self, interaction: Interaction) -> None:
        await self._graph.record_interaction(interaction)

        cache_prefix = f"recs:{interaction.user_id}:"
        await self._cache.delete_pattern(cache_prefix)
        logger.info(
            "interaction_recorded_cache_invalidated",
            user_id=str(interaction.user_id),
        )

    @staticmethod
    def _build_cache_key(user_id: UUID, request: RecommendationRequest) -> str:
        genre_str = ",".join(sorted(g.value for g in request.genres)) if request.genres else ""
        excluded_str = ",".join(sorted(str(e) for e in request.exclude_track_ids))
        return (
            f"recs:{user_id}:{request.strategy}:"
            f"{request.limit}:{request.min_popularity:.2f}:"
            f"{genre_str}:{excluded_str}"
        )

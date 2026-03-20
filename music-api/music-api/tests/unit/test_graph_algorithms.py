from __future__ import annotations

from uuid import uuid4

import pytest

from app.domain.models import (
    Genre,
    Interaction,
    InteractionType,
    Track,
    TrackCreate,
    User,
    UserCreate,
)
from app.graph.algorithms import GraphAlgorithms
from app.graph.engine import MusicGraph
from app.repositories.base import InMemoryTrackRepository, InMemoryUserRepository
from tests.conftest import make_track_create, make_user_create


async def _populate_graph(
    graph: MusicGraph,
    user_repo: InMemoryUserRepository,
    track_repo: InMemoryTrackRepository,
    n_users: int = 5,
    n_tracks: int = 10,
) -> tuple[list[User], list[Track]]:
    users = []
    for _ in range(n_users):
        u = await user_repo.create(make_user_create())
        await graph.add_user(u)
        users.append(u)

    tracks = []
    for i in range(n_tracks):
        genre = [Genre.POP, Genre.ROCK, Genre.JAZZ][i % 3]
        t = await track_repo.create(make_track_create(genres=[genre]))
        await graph.add_track(t)
        tracks.append(t)

    return users, tracks


async def _add_interactions(
    graph: MusicGraph,
    user: User,
    tracks: list[Track],
    types: list[InteractionType] | None = None,
) -> None:
    types = types or [InteractionType.PLAY, InteractionType.LIKE]
    for i, track in enumerate(tracks):
        interaction = Interaction(
            user_id=user.id,
            track_id=track.id,
            interaction_type=types[i % len(types)],
            play_duration_ms=25_000,
        )
        await graph.record_interaction(interaction)


class TestGraphEngine:
    @pytest.mark.asyncio
    async def test_add_user_creates_node(self, music_graph, user_repo):
        user = await user_repo.create(make_user_create())
        await music_graph.add_user(user)
        stats = music_graph.stats
        assert stats["user_nodes"] >= 1

    @pytest.mark.asyncio
    async def test_add_track_creates_node_and_edges(self, music_graph, track_repo):
        track = await track_repo.create(make_track_create(genres=[Genre.POP, Genre.ROCK]))
        await music_graph.add_track(track)
        stats = music_graph.stats
        assert stats["track_nodes"] >= 1
        assert stats["genre_nodes"] >= 2

    @pytest.mark.asyncio
    async def test_record_interaction_creates_edge(self, music_graph, user_repo, track_repo):
        user = await user_repo.create(make_user_create())
        track = await track_repo.create(make_track_create())
        await music_graph.add_user(user)
        await music_graph.add_track(track)

        interaction = Interaction(
            user_id=user.id,
            track_id=track.id,
            interaction_type=InteractionType.LIKE,
        )
        await music_graph.record_interaction(interaction)

        user_tracks = music_graph.get_user_tracks(user.id)
        assert len(user_tracks) == 1
        _, weight = user_tracks[0]
        assert weight == pytest.approx(3.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_multiple_interactions_accumulate_weight(
        self, music_graph, user_repo, track_repo
    ):
        user = await user_repo.create(make_user_create())
        track = await track_repo.create(make_track_create())
        await music_graph.add_user(user)
        await music_graph.add_track(track)

        for _ in range(3):
            await music_graph.record_interaction(
                Interaction(
                    user_id=user.id,
                    track_id=track.id,
                    interaction_type=InteractionType.PLAY,
                    play_duration_ms=30_000,
                )
            )

        user_tracks = music_graph.get_user_tracks(user.id)
        _, weight = user_tracks[0]
        assert weight > 1.0

    @pytest.mark.asyncio
    async def test_stats_reflect_graph_state(self, music_graph, user_repo, track_repo):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 3, 5)
        stats = music_graph.stats
        assert stats["user_nodes"] == 3
        assert stats["track_nodes"] == 5
        assert stats["total_nodes"] > 8


class TestPageRank:
    @pytest.mark.asyncio
    async def test_pagerank_returns_scored_nodes(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 3, 8)
        user = users[0]
        await _add_interactions(music_graph, user, tracks[:4])

        algo = GraphAlgorithms(music_graph)
        results = algo.personalized_pagerank(user.id, top_n=5)

        assert isinstance(results, list)
        for node in results:
            assert node.score >= 0
            assert node.entity_id

    @pytest.mark.asyncio
    async def test_pagerank_excludes_listened_tracks(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 3, 10)
        user = users[0]
        listened = tracks[:5]
        await _add_interactions(music_graph, user, listened)

        listened_ids = {str(t.id) for t in listened}

        algo = GraphAlgorithms(music_graph)
        results = algo.personalized_pagerank(user.id, top_n=20)

        for node in results:
            assert node.entity_id not in listened_ids

    @pytest.mark.asyncio
    async def test_pagerank_scores_are_sorted_descending(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 4, 12)
        user = users[0]
        await _add_interactions(music_graph, user, tracks[:6])

        algo = GraphAlgorithms(music_graph)
        results = algo.personalized_pagerank(user.id, top_n=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestContentBased:
    @pytest.mark.asyncio
    async def test_content_similarity_returns_results(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 2, 8)

        algo = GraphAlgorithms(music_graph)
        results = algo.content_based_similarity(
            seed_track_ids=[tracks[0].id, tracks[1].id],
            top_n=5,
        )
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_content_similarity_excludes_seeds(
        self, music_graph, user_repo, track_repo
    ):
        _, tracks = await _populate_graph(music_graph, user_repo, track_repo, 2, 8)
        seeds = [tracks[0].id, tracks[1].id]
        seed_ids = {str(s) for s in seeds}

        algo = GraphAlgorithms(music_graph)
        results = algo.content_based_similarity(seed_track_ids=seeds, top_n=10)

        for node in results:
            assert node.entity_id not in seed_ids

    @pytest.mark.asyncio
    async def test_content_scores_between_0_and_1(
        self, music_graph, user_repo, track_repo
    ):
        _, tracks = await _populate_graph(music_graph, user_repo, track_repo, 2, 10)

        algo = GraphAlgorithms(music_graph)
        results = algo.content_based_similarity([tracks[0].id], top_n=10)

        for node in results:
            assert -0.01 <= node.score <= 1.01


class TestCollaborativeFiltering:
    @pytest.mark.asyncio
    async def test_svd_returns_results_with_sufficient_data(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 5, 15)

        for i, user in enumerate(users):
            await _add_interactions(music_graph, user, tracks[i : i + 6])

        algo = GraphAlgorithms(music_graph)
        results = algo.collaborative_filtering(users[0].id, top_n=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_svd_returns_empty_with_insufficient_data(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 1, 2)

        algo = GraphAlgorithms(music_graph)
        results = algo.collaborative_filtering(users[0].id, top_n=5)
        assert results == []


class TestHybrid:
    @pytest.mark.asyncio
    async def test_hybrid_combines_all_sources(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 5, 20)
        user = users[0]
        for u in users:
            await _add_interactions(music_graph, u, tracks[: 10])

        algo = GraphAlgorithms(music_graph)
        results = algo.hybrid_recommendations(user.id, top_n=10)

        assert len(results) <= 10
        assert all(r.score >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_scores_sorted_descending(
        self, music_graph, user_repo, track_repo
    ):
        users, tracks = await _populate_graph(music_graph, user_repo, track_repo, 5, 20)
        for u in users:
            await _add_interactions(music_graph, u, tracks[:10])

        algo = GraphAlgorithms(music_graph)
        results = algo.hybrid_recommendations(users[0].id, top_n=15)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

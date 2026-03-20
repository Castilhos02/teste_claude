from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import networkx as nx

from app.core.config import get_settings
from app.core.exceptions import GraphException
from app.core.logging import get_logger
from app.domain.models import Interaction, NodeType, Track, User

logger = get_logger(__name__)
settings = get_settings()


class MusicGraph:
    """
    Grafo bipartido e multipartido que representa relações musicais.

    Nós:
        - user:{uuid}   → usuários
        - track:{uuid}  → músicas
        - artist:{uuid} → artistas
        - genre:{name}  → gêneros

    Arestas e pesos:
        - user → track  : peso da interação (play, like, skip, ...)
        - track → artist: pertencimento
        - track → genre : pertencimento
        - user → genre  : afinidade calculada
        - artist → genre: estilo
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = asyncio.Lock()

    @staticmethod
    def _node_id(node_type: NodeType, entity_id: str | UUID) -> str:
        return f"{node_type}:{entity_id}"

    # ── Node operations ───────────────────────────────────────────────

    async def add_user(self, user: User) -> None:
        async with self._lock:
            node_id = self._node_id(NodeType.USER, user.id)
            self._graph.add_node(
                node_id,
                type=NodeType.USER,
                entity_id=str(user.id),
                username=user.username,
                preferred_genres=[g.value for g in user.preferred_genres],
                embedding=user.embedding,
            )
            for genre in user.preferred_genres:
                await self._ensure_genre_node(genre.value)
                self._graph.add_edge(
                    node_id,
                    self._node_id(NodeType.GENRE, genre.value),
                    weight=1.0,
                    edge_type="affinity",
                )
        logger.info("user_added_to_graph", user_id=str(user.id))

    async def add_track(self, track: Track) -> None:
        async with self._lock:
            node_id = self._node_id(NodeType.TRACK, track.id)
            self._graph.add_node(
                node_id,
                type=NodeType.TRACK,
                entity_id=str(track.id),
                title=track.title,
                popularity=track.popularity,
                feature_vector=track.feature_vector,
                embedding=track.embedding,
            )
            artist_node = self._node_id(NodeType.ARTIST, track.artist_id)
            if not self._graph.has_node(artist_node):
                self._graph.add_node(
                    artist_node,
                    type=NodeType.ARTIST,
                    entity_id=str(track.artist_id),
                )
            self._graph.add_edge(
                node_id, artist_node, weight=1.0, edge_type="by_artist"
            )
            for genre in track.genres:
                await self._ensure_genre_node(genre.value)
                self._graph.add_edge(
                    node_id,
                    self._node_id(NodeType.GENRE, genre.value),
                    weight=1.0,
                    edge_type="belongs_to_genre",
                )
        logger.info("track_added_to_graph", track_id=str(track.id))

    async def _ensure_genre_node(self, genre_name: str) -> None:
        node_id = self._node_id(NodeType.GENRE, genre_name)
        if not self._graph.has_node(node_id):
            self._graph.add_node(node_id, type=NodeType.GENRE, name=genre_name)

    # ── Interaction (edge) operations ─────────────────────────────────

    async def record_interaction(self, interaction: Interaction) -> None:
        async with self._lock:
            user_node = self._node_id(NodeType.USER, interaction.user_id)
            track_node = self._node_id(NodeType.TRACK, interaction.track_id)

            if not self._graph.has_node(user_node):
                raise GraphException(
                    f"Usuário {interaction.user_id} não encontrado no grafo"
                )
            if not self._graph.has_node(track_node):
                raise GraphException(
                    f"Track {interaction.track_id} não encontrado no grafo"
                )

            if self._graph.has_edge(user_node, track_node):
                current = self._graph[user_node][track_node].get("weight", 0.0)
                new_weight = current + interaction.weight
                self._graph[user_node][track_node]["weight"] = new_weight
                self._graph[user_node][track_node]["interaction_count"] = (
                    self._graph[user_node][track_node].get("interaction_count", 0) + 1
                )
            else:
                self._graph.add_edge(
                    user_node,
                    track_node,
                    weight=interaction.weight,
                    interaction_count=1,
                    edge_type="listened",
                    last_interaction=interaction.interaction_type.value,
                )

            await self._update_genre_affinity(
                user_node,
                track_node,
                interaction.weight,
            )

        logger.info(
            "interaction_recorded",
            user_id=str(interaction.user_id),
            track_id=str(interaction.track_id),
            type=interaction.interaction_type,
            weight=interaction.weight,
        )

    async def _update_genre_affinity(
        self, user_node: str, track_node: str, delta: float
    ) -> None:
        for _, genre_node, data in self._graph.out_edges(track_node, data=True):
            if data.get("edge_type") == "belongs_to_genre":
                if self._graph.has_edge(user_node, genre_node):
                    self._graph[user_node][genre_node]["weight"] = max(
                        0.0,
                        self._graph[user_node][genre_node]["weight"] + delta * 0.1,
                    )
                else:
                    self._graph.add_edge(
                        user_node,
                        genre_node,
                        weight=max(0.0, delta * 0.1),
                        edge_type="affinity",
                    )

    # ── Queries ───────────────────────────────────────────────────────

    def get_user_tracks(
        self, user_id: UUID, min_weight: float = 0.0
    ) -> list[tuple[str, float]]:
        user_node = self._node_id(NodeType.USER, user_id)
        result = []
        for _, target, data in self._graph.out_edges(user_node, data=True):
            if (
                data.get("edge_type") == "listened"
                and data.get("weight", 0.0) >= min_weight
            ):
                result.append((target, data["weight"]))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def get_track_node_data(self, track_id: UUID) -> dict[str, Any] | None:
        node_id = self._node_id(NodeType.TRACK, track_id)
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    def get_all_track_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (n, d)
            for n, d in self._graph.nodes(data=True)
            if d.get("type") == NodeType.TRACK
        ]

    def get_all_user_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (n, d)
            for n, d in self._graph.nodes(data=True)
            if d.get("type") == NodeType.USER
        ]

    def subgraph_for_user(self, user_id: UUID, depth: int = 2) -> nx.DiGraph:
        user_node = self._node_id(NodeType.USER, user_id)
        if not self._graph.has_node(user_node):
            raise GraphException(f"Usuário {user_id} não encontrado no grafo")
        nodes = nx.ego_graph(self._graph, user_node, radius=depth).nodes()
        return self._graph.subgraph(nodes).copy()

    @property
    def stats(self) -> dict[str, int]:
        by_type: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = str(data.get("type", "unknown"))
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            **{f"{k}_nodes": v for k, v in by_type.items()},
        }

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

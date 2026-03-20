from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.core.config import get_settings
from app.core.exceptions import RecommendationException
from app.core.logging import get_logger
from app.domain.models import NodeType

if TYPE_CHECKING:
    from app.graph.engine import MusicGraph

logger = get_logger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class ScoredNode:
    node_id: str
    entity_id: str
    score: float


class GraphAlgorithms:
    """
    Algoritmos de recomendação que operam sobre o MusicGraph.

    Estratégias implementadas:
    1. PageRank personalizado (Personalized PageRank)
    2. Similaridade de conteúdo via cosine (audio features)
    3. Collaborative Filtering via fatorização de matriz (SVD)
    4. Hybrid scoring com pesos configuráveis
    """

    def __init__(self, music_graph: MusicGraph) -> None:
        self._g = music_graph

    # ── 1. Personalized PageRank ───────────────────────────────────────

    def personalized_pagerank(
        self,
        user_id: UUID,
        top_n: int = 20,
        alpha: float | None = None,
    ) -> list[ScoredNode]:
        """
        Executa PageRank personalizado a partir do nó do usuário.
        O vetor de personalização dá mais peso às tracks já ouvidas,
        o que propaga relevância para tracks vizinhas no grafo.
        """
        alpha = alpha or settings.pagerank_alpha
        user_node = f"{NodeType.USER}:{user_id}"
        subgraph = self._g.subgraph_for_user(user_id, depth=3)

        if subgraph.number_of_nodes() < 3:
            return []

        # Monta o vetor de personalização
        listened = {
            track_node: weight
            for track_node, weight in self._g.get_user_tracks(user_id, min_weight=0.0)
        }

        personalization: dict[str, float] = {}
        total = max(sum(listened.values()), 1.0)
        for node in subgraph.nodes():
            if node == user_node:
                personalization[node] = 0.1
            elif node in listened:
                personalization[node] = listened[node] / total * 0.9
            else:
                personalization[node] = 0.001

        p_sum = sum(personalization.values())
        personalization = {k: v / p_sum for k, v in personalization.items()}

        try:
            scores = nx.pagerank(
                subgraph,
                alpha=alpha,
                personalization=personalization,
                max_iter=200,
                tol=1e-6,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence as exc:
            logger.warning("pagerank_failed_convergence", user_id=str(user_id))
            raise RecommendationException("PageRank não convergiu") from exc

        already_listened = set(listened.keys())
        results = []
        for node, score in scores.items():
            if (
                node.startswith(f"{NodeType.TRACK}:")
                and node not in already_listened
            ):
                entity_id = node.split(":", 1)[1]
                results.append(ScoredNode(node_id=node, entity_id=entity_id, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # ── 2. Content-based: cosine similarity on audio features ─────────

    def content_based_similarity(
        self,
        seed_track_ids: list[UUID],
        candidate_track_ids: list[UUID] | None = None,
        top_n: int = 20,
    ) -> list[ScoredNode]:
        """
        Calcula similaridade cosine entre vetores de audio features.
        Seed tracks = tracks que o usuário gostou.
        Candidates = todo o catálogo (ou subconjunto filtrado).
        """
        all_tracks = self._g.get_all_track_nodes()

        track_index: dict[str, int] = {}
        feature_rows: list[list[float]] = []

        for i, (node_id, data) in enumerate(all_tracks):
            fv = data.get("feature_vector")
            if fv and len(fv) >= 7:
                track_index[node_id] = i
                feature_rows.append(fv)

        if len(feature_rows) < 2:
            return []

        matrix = np.array(feature_rows, dtype=np.float32)
        matrix_norm = normalize(matrix, norm="l2")

        seed_indices = []
        for sid in seed_track_ids:
            node_id = f"{NodeType.TRACK}:{sid}"
            if node_id in track_index:
                seed_indices.append(track_index[node_id])

        if not seed_indices:
            return []

        seed_matrix = matrix_norm[seed_indices]
        seed_profile = seed_matrix.mean(axis=0, keepdims=True)
        similarities = cosine_similarity(seed_profile, matrix_norm)[0]

        excluded = {f"{NodeType.TRACK}:{sid}" for sid in seed_track_ids}
        if candidate_track_ids is not None:
            allowed = {f"{NodeType.TRACK}:{cid}" for cid in candidate_track_ids}
        else:
            allowed = None

        results = []
        for node_id, idx in track_index.items():
            if node_id in excluded:
                continue
            if allowed is not None and node_id not in allowed:
                continue
            entity_id = node_id.split(":", 1)[1]
            results.append(
                ScoredNode(node_id=node_id, entity_id=entity_id, score=float(similarities[idx]))
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # ── 3. Collaborative Filtering via SVD ────────────────────────────

    def collaborative_filtering(
        self,
        user_id: UUID,
        top_n: int = 20,
        n_components: int = 50,
    ) -> list[ScoredNode]:
        """
        Fatorização de matriz usuário-item usando SVD truncado.
        Reconstrói preferências latentes e pontua tracks não ouvidas.
        """
        users = self._g.get_all_user_nodes()
        tracks = self._g.get_all_track_nodes()

        if len(users) < 3 or len(tracks) < 3:
            return []

        user_ids = [d["entity_id"] for _, d in users]
        track_nodes = [node_id for node_id, _ in tracks]
        track_ids = [d["entity_id"] for _, d in tracks]

        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        track_to_idx = {tnid: i for i, tnid in enumerate(track_nodes)}

        R = np.zeros((len(user_ids), len(track_ids)), dtype=np.float32)
        for uid_str, i in user_to_idx.items():
            uid = UUID(uid_str)
            for track_node, weight in self._g.get_user_tracks(uid, min_weight=-10.0):
                if track_node in track_to_idx:
                    j = track_to_idx[track_node]
                    R[i, j] = float(weight)

        target_user_str = str(user_id)
        if target_user_str not in user_to_idx:
            return []

        target_idx = user_to_idx[target_user_str]

        effective_components = min(n_components, min(R.shape) - 1)
        if effective_components < 1:
            return []

        try:
            svd = TruncatedSVD(n_components=effective_components, random_state=42)
            U = svd.fit_transform(R)
            Vt = svd.components_
            R_hat = U @ Vt
        except Exception as exc:
            logger.warning("svd_failed", error=str(exc))
            raise RecommendationException("SVD falhou") from exc

        already_listened = {
            tn for tn, w in self._g.get_user_tracks(user_id, min_weight=0.0)
        }

        predicted_scores = R_hat[target_idx]
        results = []
        for j, track_node in enumerate(track_nodes):
            if track_node in already_listened:
                continue
            entity_id = track_ids[j]
            results.append(
                ScoredNode(
                    node_id=track_node,
                    entity_id=entity_id,
                    score=float(predicted_scores[j]),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # ── 4. Embedding-based similarity ─────────────────────────────────

    def embedding_similarity(
        self,
        user_id: UUID,
        top_n: int = 20,
    ) -> list[ScoredNode]:
        """
        Similaridade entre o embedding do usuário e embeddings das tracks.
        Embeddings podem vir de modelos externos (Word2Vec, BERT, etc.)
        ou serem aprendidos internamente via SVD de interações.
        """
        user_node = f"{NodeType.USER}:{user_id}"
        if not self._g.graph.has_node(user_node):
            return []

        user_data = self._g.graph.nodes[user_node]
        user_embedding = user_data.get("embedding")
        if user_embedding is None:
            return []

        user_vec = np.array(user_embedding, dtype=np.float32).reshape(1, -1)
        user_vec = normalize(user_vec)

        already_listened = {
            tn for tn, _ in self._g.get_user_tracks(user_id, min_weight=0.0)
        }

        track_nodes_with_embeddings = [
            (node_id, data)
            for node_id, data in self._g.get_all_track_nodes()
            if data.get("embedding") is not None and node_id not in already_listened
        ]

        if not track_nodes_with_embeddings:
            return []

        track_matrix = np.array(
            [data["embedding"] for _, data in track_nodes_with_embeddings],
            dtype=np.float32,
        )
        track_matrix = normalize(track_matrix)

        scores = cosine_similarity(user_vec, track_matrix)[0]
        results = [
            ScoredNode(
                node_id=node_id,
                entity_id=data["entity_id"],
                score=float(scores[i]),
            )
            for i, (node_id, data) in enumerate(track_nodes_with_embeddings)
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # ── 5. Hybrid scoring ─────────────────────────────────────────────

    def hybrid_recommendations(
        self,
        user_id: UUID,
        top_n: int = 20,
        weights: dict[str, float] | None = None,
    ) -> list[ScoredNode]:
        """
        Combina todos os algoritmos com pesos ponderados.
        Normaliza cada ranking individualmente antes de combinar.
        """
        w = weights or {
            "pagerank": 0.30,
            "collaborative": 0.35,
            "content": 0.25,
            "embedding": 0.10,
        }

        fetch_n = top_n * 3

        sources: dict[str, list[ScoredNode]] = {}

        try:
            sources["pagerank"] = self.personalized_pagerank(user_id, fetch_n)
        except RecommendationException:
            sources["pagerank"] = []

        try:
            sources["collaborative"] = self.collaborative_filtering(user_id, fetch_n)
        except RecommendationException:
            sources["collaborative"] = []

        user_liked = [
            UUID(track_node.split(":", 1)[1])
            for track_node, weight in self._g.get_user_tracks(user_id, min_weight=1.0)
        ][:20]
        if user_liked:
            try:
                sources["content"] = self.content_based_similarity(user_liked, top_n=fetch_n)
            except Exception:
                sources["content"] = []
        else:
            sources["content"] = []

        try:
            sources["embedding"] = self.embedding_similarity(user_id, fetch_n)
        except Exception:
            sources["embedding"] = []

        combined: dict[str, float] = {}
        for source_name, nodes in sources.items():
            if not nodes:
                continue
            max_score = max((n.score for n in nodes), default=1.0)
            if max_score == 0:
                max_score = 1.0
            source_weight = w.get(source_name, 0.0)
            for rank, node in enumerate(nodes):
                normalized = (node.score / max_score) * source_weight
                combined[node.entity_id] = combined.get(node.entity_id, 0.0) + normalized

        if not combined:
            return []

        results = []
        for entity_id, score in sorted(combined.items(), key=lambda x: x[1], reverse=True):
            node_id = f"{NodeType.TRACK}:{entity_id}"
            results.append(ScoredNode(node_id=node_id, entity_id=entity_id, score=score))

        return results[:top_n]

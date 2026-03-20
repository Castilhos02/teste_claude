from fastapi import APIRouter

from app.api.dependencies import CurrentUser, RecommendationSvc
from app.domain.models import (
    Interaction,
    InteractionCreate,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    current_user: CurrentUser,
    svc: RecommendationSvc,
) -> RecommendationResponse:
    """
    Gera recomendações musicais personalizadas.

    Estratégias disponíveis:
    - `hybrid` (padrão): combina todos os algoritmos
    - `graph_pagerank`: PageRank personalizado no grafo
    - `content_based`: similaridade por audio features
    - `collaborative_filtering`: fatorização de matriz SVD
    - `similar_users`: similaridade de embeddings de usuários
    """
    return await svc.get_recommendations(current_user.id, request)


@router.post("/interactions", status_code=204)
async def record_interaction(
    payload: InteractionCreate,
    current_user: CurrentUser,
    svc: RecommendationSvc,
) -> None:
    """
    Registra uma interação do usuário com uma música.
    Invalida o cache de recomendações do usuário automaticamente.
    """
    interaction = Interaction(
        **payload.model_dump(exclude={"user_id"}),
        user_id=current_user.id,
    )
    await svc.record_interaction(interaction)

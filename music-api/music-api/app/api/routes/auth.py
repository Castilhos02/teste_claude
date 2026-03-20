from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.api.dependencies import UserRepo
from app.core.exceptions import ConflictException, UnauthorizedException
from app.core.security import TokenPair, create_token_pair, verify_password
from app.domain.models import UserCreate, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterResponse(BaseModel):
    user: UserPublic
    tokens: TokenPair


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    payload: UserCreate,
    user_repo: UserRepo,
) -> RegisterResponse:
    if await user_repo.get_by_email(payload.email):
        raise ConflictException("E-mail já cadastrado")
    if await user_repo.get_by_username(payload.username):
        raise ConflictException("Username já em uso")

    user = await user_repo.create(payload)
    tokens = create_token_pair(user.id)

    return RegisterResponse(
        user=UserPublic(
            id=user.id,
            username=user.username,
            email=user.email,
            display_name=user.display_name,
            preferred_genres=user.preferred_genres,
            created_at=user.created_at,
        ),
        tokens=tokens,
    )


@router.post("/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    user_repo: UserRepo,
) -> TokenPair:
    user = await user_repo.get_by_email(payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise UnauthorizedException("Credenciais inválidas")
    if not user.is_active:
        raise UnauthorizedException("Conta desativada")

    return create_token_pair(user.id)

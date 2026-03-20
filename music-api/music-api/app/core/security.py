from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedException

settings = get_settings()

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
    iat: datetime
    type: str
    jti: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def _create_token(
    subject: str | UUID,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    import uuid

    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + expires_delta,
        "type": token_type,
        "jti": str(uuid.uuid4()),
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def create_token_pair(user_id: str | UUID, roles: list[str] | None = None) -> TokenPair:
    access_exp = timedelta(minutes=settings.access_token_expire_minutes)
    refresh_exp = timedelta(days=settings.refresh_token_expire_days)

    claims = {"roles": roles or []}

    return TokenPair(
        access_token=_create_token(user_id, "access", access_exp, claims),
        refresh_token=_create_token(user_id, "refresh", refresh_exp),
        expires_in=int(access_exp.total_seconds()),
    )


def decode_token(token: str, expected_type: str = "access") -> TokenPayload:
    try:
        raw = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise UnauthorizedException("Token inválido ou expirado") from exc

    payload = TokenPayload(**raw)

    if payload.type != expected_type:
        raise UnauthorizedException(f"Token type inválido: esperado '{expected_type}'")

    return payload

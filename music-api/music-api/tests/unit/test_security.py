from uuid import uuid4

import pytest
from jose import jwt

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedException
from app.core.security import (
    create_token_pair,
    decode_token,
    hash_password,
    verify_password,
)

settings = get_settings()


class TestPasswordHashing:
    def test_hash_is_different_from_plain(self):
        plain = "MinhaSenh@123"
        hashed = hash_password(plain)
        assert hashed != plain

    def test_verify_correct_password(self):
        plain = "MinhaSenh@123"
        assert verify_password(plain, hash_password(plain)) is True

    def test_reject_wrong_password(self):
        assert verify_password("errada", hash_password("certa")) is False

    def test_same_plain_generates_different_hashes(self):
        plain = "senha123A"
        h1 = hash_password(plain)
        h2 = hash_password(plain)
        assert h1 != h2


class TestJWT:
    def test_create_token_pair_returns_both_tokens(self):
        pair = create_token_pair(uuid4())
        assert pair.access_token
        assert pair.refresh_token
        assert pair.token_type == "bearer"
        assert pair.expires_in > 0

    def test_decode_valid_access_token(self):
        uid = uuid4()
        pair = create_token_pair(uid)
        payload = decode_token(pair.access_token, "access")
        assert payload.sub == str(uid)
        assert payload.type == "access"

    def test_decode_wrong_type_raises(self):
        pair = create_token_pair(uuid4())
        with pytest.raises(UnauthorizedException):
            decode_token(pair.access_token, "refresh")

    def test_tampered_token_raises(self):
        pair = create_token_pair(uuid4())
        tampered = pair.access_token[:-5] + "XXXXX"
        with pytest.raises(UnauthorizedException):
            decode_token(tampered)

    def test_token_contains_roles(self):
        uid = uuid4()
        pair = create_token_pair(uid, roles=["admin", "user"])
        raw = jwt.decode(
            pair.access_token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        assert "admin" in raw["roles"]

    def test_jti_is_unique_per_token(self):
        uid = uuid4()
        p1 = create_token_pair(uid)
        p2 = create_token_pair(uid)
        raw1 = jwt.decode(
            p1.access_token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        raw2 = jwt.decode(
            p2.access_token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        assert raw1["jti"] != raw2["jti"]

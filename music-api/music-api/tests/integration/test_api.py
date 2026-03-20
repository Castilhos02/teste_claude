from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient

from tests.conftest import make_track_create, make_user_create


class TestAuth:
    @pytest.mark.asyncio
    async def test_register_returns_user_and_tokens(self, client: AsyncClient):
        payload = make_user_create()
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "username": payload.username,
                "email": payload.email,
                "display_name": payload.display_name,
                "password": "Senha123",
                "confirm_password": "Senha123",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "user" in data
        assert "tokens" in data
        assert data["tokens"]["token_type"] == "bearer"
        assert data["user"]["email"] == payload.email

    @pytest.mark.asyncio
    async def test_register_duplicate_email_returns_409(self, client: AsyncClient):
        payload = make_user_create()
        body = {
            "username": payload.username,
            "email": payload.email,
            "display_name": payload.display_name,
            "password": "Senha123",
            "confirm_password": "Senha123",
        }
        await client.post("/api/v1/auth/register", json=body)
        resp = await client.post("/api/v1/auth/register", json=body)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_login_returns_token_pair(self, client: AsyncClient, registered_user):
        email = registered_user["user"]["email"]
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "Senha123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(
        self, client: AsyncClient, registered_user
    ):
        email = registered_user["user"]["email"]
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "errada123"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_weak_password_returns_422(self, client: AsyncClient):
        payload = make_user_create()
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "username": payload.username,
                "email": payload.email,
                "display_name": payload.display_name,
                "password": "fraca",
                "confirm_password": "fraca",
            },
        )
        assert resp.status_code == 422


class TestTracks:
    @pytest.mark.asyncio
    async def test_create_track_requires_auth(self, client: AsyncClient):
        payload = make_track_create()
        resp = await client.post(
            "/api/v1/tracks",
            json=payload.model_dump(mode="json"),
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_create_and_retrieve_track(
        self, client: AsyncClient, auth_headers: dict
    ):
        payload = make_track_create(title="Test Track Alpha")
        resp = await client.post(
            "/api/v1/tracks",
            json=payload.model_dump(mode="json"),
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Track Alpha"
        track_id = data["id"]

        get_resp = await client.get(
            f"/api/v1/tracks/{track_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == track_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_track_returns_404(
        self, client: AsyncClient, auth_headers: dict
    ):
        resp = await client.get(
            f"/api/v1/tracks/{uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_tracks(self, client: AsyncClient, auth_headers: dict):
        for i in range(3):
            await client.post(
                "/api/v1/tracks",
                json=make_track_create(title=f"List Track {i}").model_dump(mode="json"),
                headers=auth_headers,
            )
        resp = await client.get("/api/v1/tracks?limit=10", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestRecommendations:
    @pytest.mark.asyncio
    async def test_recommendations_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/recommendations",
            json={"limit": 10},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_recommendations_returns_response_schema(
        self, client: AsyncClient, auth_headers: dict
    ):
        resp = await client.post(
            "/api/v1/recommendations",
            json={"limit": 5, "strategy": "hybrid"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "algorithm_used" in data
        assert isinstance(data["recommendations"], list)

    @pytest.mark.asyncio
    async def test_recommendations_with_tracks_in_graph(
        self, client: AsyncClient, auth_headers: dict
    ):
        track_ids = []
        for i in range(5):
            t_resp = await client.post(
                "/api/v1/tracks",
                json=make_track_create(title=f"Reco Track {i}").model_dump(mode="json"),
                headers=auth_headers,
            )
            track_ids.append(t_resp.json()["id"])

        await client.post(
            "/api/v1/recommendations/interactions",
            json={
                "user_id": str(uuid4()),
                "track_id": track_ids[0],
                "interaction_type": "like",
            },
            headers=auth_headers,
        )

        resp = await client.post(
            "/api/v1/recommendations",
            json={"limit": 5, "strategy": "graph_pagerank"},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_record_interaction(
        self, client: AsyncClient, auth_headers: dict
    ):
        t_resp = await client.post(
            "/api/v1/tracks",
            json=make_track_create().model_dump(mode="json"),
            headers=auth_headers,
        )
        track_id = t_resp.json()["id"]

        resp = await client.post(
            "/api/v1/recommendations/interactions",
            json={
                "user_id": str(uuid4()),
                "track_id": track_id,
                "interaction_type": "play",
                "play_duration_ms": 25000,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_recommendations_limit_respected(
        self, client: AsyncClient, auth_headers: dict
    ):
        resp = await client.post(
            "/api/v1/recommendations",
            json={"limit": 3},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) <= 3


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "graph_stats" in data
        assert "dependencies" in data

    @pytest.mark.asyncio
    async def test_readiness_probe(self, client: AsyncClient):
        resp = await client.get("/health/ready")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_liveness_probe(self, client: AsyncClient):
        resp = await client.get("/health/live")
        assert resp.status_code == 200

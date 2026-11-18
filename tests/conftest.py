from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.helpers import auth_header

__all__ = ["auth_header"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        jwt_secret="test-secret-valido-para-firmar-jwt",
        storage_backend="memory",
        chroma_backend="memory",
        embedding_provider="local",
        llm_provider="local",
        openai_api_key=None,
        gemini_api_key=None,
        google_api_key=None,
        firestore_project_id=None,
        chroma_path=str(tmp_path / "chroma"),
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def auth_header(app, business_id: str) -> dict[str, str]:
    token = mint_token(business_id, app.state.settings)
    return {"Authorization": f"Bearer {token}"}

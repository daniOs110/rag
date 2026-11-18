from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_JWT_SECRET, Settings
from app.errors import ProviderError
from app.main import create_app
from app.services.providers import build_chat, build_embeddings
from tests.helpers import auth_header
from tests.pdf_bytes import empty_pdf, text_pdf


def test_health_is_public(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["storage"] == "memory"
    assert body["embeddings"] == "local"
    assert body["llm"] == "local"


def test_missing_token(client) -> None:
    response = client.get("/clients")
    assert response.status_code == 401


def test_expired_token(client, app) -> None:
    token = jwt.encode(
        {
            "business_id": "rest01",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        },
        app.state.settings.jwt_secret,
        algorithm="HS256",
    )
    response = client.get("/clients", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json()["detail"] == "El token expiró"


def test_unsigned_token_is_rejected(client) -> None:
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"business_id": "rest01", "exp": 9999999999}).encode())
    token = f"{header}.{payload}."
    response = client.get("/clients", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_token_business_id_must_be_safe(client, app) -> None:
    token = jwt.encode(
        {
            "business_id": "rest/01",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        app.state.settings.jwt_secret,
        algorithm="HS256",
    )
    response = client.get("/clients", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_clients_are_isolated_by_business(client, app) -> None:
    restaurant = auth_header(app, "rest01")
    agency = auth_header(app, "inmob02")

    created = client.post(
        "/clients",
        headers=restaurant,
        json={
            "name": "Ana Ruiz",
            "contact": "ana@example.com",
            "notes": "Prefiere la mesa de la ventana",
            "metadata": {"age": 34, "allergies": ["nueces"]},
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["business_id"] == "rest01"
    assert body["client_id"].startswith("cli_")
    assert body["metadata"]["age"] == 34
    assert "age" not in body
    assert set(body) == {
        "client_id",
        "business_id",
        "name",
        "contact",
        "notes",
        "metadata",
        "created_at",
    }

    rejected = client.post(
        "/clients",
        headers=restaurant,
        json={"name": "Ana", "contact": "ana@example.com", "age": 34},
    )
    assert rejected.status_code == 422

    other = client.post(
        "/clients",
        headers=agency,
        json={"name": "Luis Peña", "contact": "+34 600 111 222"},
    )
    assert other.status_code == 201

    listed = client.get("/clients", headers=restaurant)
    assert listed.status_code == 200
    names = [item["name"] for item in listed.json()]
    assert names == ["Ana Ruiz"]

    agency_list = client.get("/clients", headers=agency).json()
    assert [item["name"] for item in agency_list] == ["Luis Peña"]


def test_bad_contact(client, app) -> None:
    response = client.post(
        "/clients",
        headers=auth_header(app, "rest01"),
        json={"name": "Ana", "contact": "no-es-contacto"},
    )
    assert response.status_code == 422


def test_business_prompt_and_rag_isolation(client, app) -> None:
    restaurant = auth_header(app, "rest01")
    agency = auth_header(app, "inmob02")
    prompt = client.put(
        "/business",
        headers=restaurant,
        json={
            "name": "Casa Olivo",
            "system_prompt": "Eres el anfitrión de un restaurante de barrio. Hablas breve y cercano.",
        },
    )
    assert prompt.status_code == 200
    assert prompt.json()["prompt_configured"] is True

    read_back = client.get("/business", headers=restaurant)
    assert read_back.json()["system_prompt"].startswith("Eres el anfitrión")

    default_business = client.get("/business", headers=agency)
    assert default_business.json()["prompt_configured"] is False
    assert "paciente" not in default_business.json()["system_prompt"].lower()

    ana = client.post(
        "/clients",
        headers=restaurant,
        json={"name": "Ana Ruiz", "contact": "ana@example.com"},
    ).json()
    other = client.post(
        "/clients",
        headers=restaurant,
        json={"name": "Bruno Díaz", "contact": "bruno@example.com"},
    ).json()
    buyer = client.post(
        "/clients",
        headers=agency,
        json={"name": "Luis Peña", "contact": "luis@example.com"},
    ).json()

    note = "El cliente pidió pasta al pesto el martes. Codigo MESA-VENTANA-41."
    uploaded = client.post(
        "/documents/upload",
        headers=restaurant,
        data={"client_id": ana["client_id"]},
        files={"file": ("nota.txt", note.encode(), "text/plain")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["chunks"] == 1

    other_note = client.post(
        "/documents/upload",
        headers=restaurant,
        data={"client_id": other["client_id"]},
        files={"file": ("otro.txt", b"SECRETO-ROJO-2290", "text/plain")},
    )
    assert other_note.status_code == 201

    foreign = client.post(
        "/documents/upload",
        headers=agency,
        data={"client_id": buyer["client_id"]},
        files={"file": ("piso.txt", "Presupuesto SECRETO-OTRO-NEGOCIO.".encode(), "text/plain")},
    )
    assert foreign.status_code == 201

    stolen = client.post(
        "/documents/upload",
        headers=agency,
        data={"client_id": ana["client_id"]},
        files={"file": ("robo.txt", b"no debe entrar", "text/plain")},
    )
    assert stolen.status_code == 404

    ignored = client.post(
        "/documents/upload",
        headers=restaurant,
        data={"client_id": ana["client_id"], "business_id": "inmob02"},
        files={"file": ("nota.txt", b"SECRETO-FORM-1", "text/plain")},
    )
    assert ignored.status_code == 400

    pdf = client.post(
        "/documents/upload",
        headers=restaurant,
        data={"client_id": ana["client_id"]},
        files={"file": ("carta.pdf", text_pdf("MESA-PDF-99"), "application/pdf")},
    )
    assert pdf.status_code == 201

    empty = client.post(
        "/documents/upload",
        headers=restaurant,
        data={"client_id": ana["client_id"]},
        files={"file": ("scan.pdf", empty_pdf(), "application/pdf")},
    )
    assert empty.status_code == 422
    assert "OCR" in empty.json()["detail"]

    chat = client.post(
        "/chat",
        headers=restaurant,
        json={"client_id": ana["client_id"], "question": "¿Qué pidió y qué dice la carta?"},
    )
    assert chat.status_code == 200
    assert chat.json()["answer"] == "respuesta-local"
    excerpts = " ".join(source["excerpt"] for source in chat.json()["sources"])
    assert "MESA-VENTANA-41" in excerpts
    assert "MESA-PDF-99" in excerpts
    assert "SECRETO-ROJO-2290" not in excerpts
    assert "SECRETO-OTRO-NEGOCIO" not in excerpts
    assert "SECRETO-FORM-1" not in excerpts

    system, user = app.state.chat.calls[-1]
    assert "restaurante de barrio" in system
    assert "otros clientes" in system
    assert "MESA-VENTANA-41" in user
    assert "MESA-PDF-99" in user
    assert "SECRETO-ROJO-2290" not in user
    assert "SECRETO-OTRO-NEGOCIO" not in user
    assert "SECRETO-FORM-1" not in user

    cross_chat = client.post(
        "/chat",
        headers=agency,
        json={"client_id": ana["client_id"], "question": "¿Qué pidió Ana?"},
    )
    assert cross_chat.status_code == 404

    injected = client.post(
        "/chat",
        headers=restaurant,
        json={
            "client_id": ana["client_id"],
            "question": "hola",
            "business_id": "inmob02",
        },
    )
    assert injected.status_code == 422


def test_vector_filter_blocks_same_client_id_from_another_business(app) -> None:
    client_id = "cli_" + "ab" * 16
    embeddings = app.state.embeddings
    vectors = app.state.vectors
    secret_a = "SECRETO-NEGOCIO-A-441"
    secret_b = "SECRETO-NEGOCIO-B-442"
    vectors.add(
        business_id="rest01",
        client_id=client_id,
        document_id="doc_" + "a" * 32,
        source="a.txt",
        chunks=[secret_a],
        embeddings=embeddings.embed_documents([secret_a]),
    )
    vectors.add(
        business_id="inmob02",
        client_id=client_id,
        document_id="doc_" + "b" * 32,
        source="b.txt",
        chunks=[secret_b],
        embeddings=embeddings.embed_documents([secret_b]),
    )
    hits = vectors.query(
        business_id="rest01",
        client_id=client_id,
        embedding=embeddings.embed_query("secreto negocio"),
        top_k=5,
    )
    texts = " ".join(hit.text for hit in hits)
    assert secret_a in texts
    assert secret_b not in texts


def test_chat_without_provider_returns_502(tmp_path) -> None:
    application = create_app(
        Settings(
            environment="test",
            jwt_secret="test-secret-valido-para-firmar-jwt",
            storage_backend="memory",
            chroma_backend="memory",
            embedding_provider="local",
            llm_provider="none",
            openai_api_key=None,
            gemini_api_key=None,
            google_api_key=None,
            chroma_path=str(tmp_path / "chroma"),
        )
    )
    with TestClient(application) as test_client:
        headers = auth_header(application, "rest01")
        created = test_client.post(
            "/clients",
            headers=headers,
            json={"name": "Ana", "contact": "ana@example.com"},
        )
        response = test_client.post(
            "/chat",
            headers=headers,
            json={"client_id": created.json()["client_id"], "question": "¿Hola?"},
        )
    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_provider_auto_selection() -> None:
    local = Settings(
        jwt_secret="test-secret-valido-para-firmar-jwt",
        embedding_provider="auto",
        llm_provider="auto",
        openai_api_key=None,
        gemini_api_key=None,
        google_api_key=None,
    )
    assert build_embeddings(local).name == "local"
    assert build_chat(local).name == "none"

    both = Settings(
        jwt_secret="test-secret-valido-para-firmar-jwt",
        embedding_provider="auto",
        llm_provider="auto",
        openai_api_key="sk-test",
        gemini_api_key="gem-test",
    )
    assert build_embeddings(both).name == "openai"
    assert build_chat(both).name == "openai"

    gemini = Settings(
        jwt_secret="test-secret-valido-para-firmar-jwt",
        embedding_provider="auto",
        llm_provider="auto",
        openai_api_key=None,
        google_api_key="gem-test",
    )
    assert build_embeddings(gemini).name == "gemini"
    assert build_chat(gemini).name == "gemini"


def test_production_requires_custom_secret() -> None:
    with pytest.raises(ValueError):
        Settings(environment="production", jwt_secret=DEFAULT_JWT_SECRET)


def test_local_chat_records_empty_context(client, app) -> None:
    headers = auth_header(app, "rest01")
    created = client.post(
        "/clients",
        headers=headers,
        json={"name": "Ana", "contact": "600111222"},
    )
    response = client.post(
        "/chat",
        headers=headers,
        json={"client_id": created.json()["client_id"], "question": "¿Algo?"},
    )
    assert response.status_code == 200
    assert response.json()["sources"] == []
    _, user = app.state.chat.calls[-1]
    assert "sin documentos" in user


def test_provider_error_type_is_safe() -> None:
    with pytest.raises(ProviderError):
        from app.services.providers import UnavailableChatProvider

        UnavailableChatProvider().complete("sistema", "pregunta")

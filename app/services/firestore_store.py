from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.store import Business, Client, utcnow


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return utcnow()


class FirestoreStore:
    """Negocios y clientes bajo businesses/{business_id}.

    El client_id se lee siempre dentro de la subcolección de ese negocio.
    Un identificador conocido de otro negocio no abre sus datos.
    """

    def __init__(self, project_id: str | None) -> None:
        from google.cloud import firestore

        self._firestore = firestore
        if project_id:
            self._db = firestore.Client(project=project_id)
        else:
            self._db = firestore.Client()

    def _business_ref(self, business_id: str):
        return self._db.collection("businesses").document(business_id)

    def _clients_ref(self, business_id: str):
        return self._business_ref(business_id).collection("clients")

    def get_business(self, business_id: str) -> Business | None:
        snapshot = self._business_ref(business_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return Business(
            business_id=business_id,
            name=str(data.get("name") or business_id),
            system_prompt=data.get("system_prompt"),
            updated_at=_as_datetime(data.get("updated_at")),
        )

    def upsert_business(self, business: Business) -> Business:
        self._business_ref(business.business_id).set(
            {
                "name": business.name,
                "system_prompt": business.system_prompt,
                "updated_at": business.updated_at,
            }
        )
        return business

    def create_client(self, client: Client) -> Client:
        self._clients_ref(client.business_id).document(client.client_id).set(
            {
                "name": client.name,
                "contact": client.contact,
                "notes": client.notes,
                "metadata": client.metadata,
                "created_at": client.created_at,
            }
        )
        return client

    def get_client(self, business_id: str, client_id: str) -> Client | None:
        snapshot = self._clients_ref(business_id).document(client_id).get()
        if not snapshot.exists:
            return None
        return self._to_client(business_id, client_id, snapshot.to_dict() or {})

    def list_clients(self, business_id: str, limit: int) -> list[Client]:
        query = (
            self._clients_ref(business_id)
            .order_by("created_at", direction=self._firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [
            self._to_client(business_id, doc.id, doc.to_dict() or {})
            for doc in query.stream()
        ]

    def _to_client(self, business_id: str, client_id: str, data: dict[str, Any]) -> Client:
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        notes = data.get("notes")
        return Client(
            client_id=client_id,
            business_id=business_id,
            name=str(data.get("name") or ""),
            contact=str(data.get("contact") or ""),
            notes=notes if isinstance(notes, str) else None,
            metadata=metadata,
            created_at=_as_datetime(data.get("created_at")),
        )

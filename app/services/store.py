from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Business:
    business_id: str
    name: str
    system_prompt: str | None
    updated_at: datetime


@dataclass
class Client:
    client_id: str
    business_id: str
    name: str
    contact: str
    notes: str | None
    metadata: dict[str, Any]
    created_at: datetime
    sequence: int = field(default=0, compare=False)


class Store(Protocol):
    def get_business(self, business_id: str) -> Business | None: ...

    def upsert_business(self, business: Business) -> Business: ...

    def create_client(self, client: Client) -> Client: ...

    def get_client(self, business_id: str, client_id: str) -> Client | None: ...

    def list_clients(self, business_id: str, limit: int) -> list[Client]: ...


class MemoryStore:
    """Almacén en proceso. Cada lectura devuelve una copia para no filtrar estado interno."""

    def __init__(self) -> None:
        self._businesses: dict[str, Business] = {}
        self._clients: dict[tuple[str, str], Client] = {}
        self._sequence = 0
        self._lock = Lock()

    def get_business(self, business_id: str) -> Business | None:
        with self._lock:
            business = self._businesses.get(business_id)
            return copy.deepcopy(business) if business else None

    def upsert_business(self, business: Business) -> Business:
        with self._lock:
            stored = copy.deepcopy(business)
            self._businesses[business.business_id] = stored
            return copy.deepcopy(stored)

    def create_client(self, client: Client) -> Client:
        with self._lock:
            self._sequence += 1
            stored = copy.deepcopy(client)
            stored.sequence = self._sequence
            self._clients[(stored.business_id, stored.client_id)] = stored
            return copy.deepcopy(stored)

    def get_client(self, business_id: str, client_id: str) -> Client | None:
        with self._lock:
            client = self._clients.get((business_id, client_id))
            return copy.deepcopy(client) if client else None

    def list_clients(self, business_id: str, limit: int) -> list[Client]:
        with self._lock:
            owned = [client for (owner, _), client in self._clients.items() if owner == business_id]
            owned.sort(key=lambda client: (client.created_at, client.sequence), reverse=True)
            return [copy.deepcopy(client) for client in owned[:limit]]

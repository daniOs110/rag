from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import decode_business_id
from app.errors import AuthError
from app.ids import valid_client_id
from app.services.store import Client

_bearer = HTTPBearer(auto_error=False)


def get_business_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Falta el token de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_business_id(credentials.credentials, request.app.state.settings)
    except AuthError as exc:
        raise HTTPException(
            status_code=401,
            detail=exc.detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_client(request: Request, business_id: str, client_id: str) -> Client:
    if not valid_client_id(client_id):
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    client = request.app.state.store.get_client(business_id, client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client

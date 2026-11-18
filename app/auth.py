from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from app.config import Settings
from app.errors import AuthError
from app.ids import valid_business_id


def decode_business_id(token: str, settings: Settings) -> str:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "business_id"]},
            leeway=30,
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("El token expiró") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Token inválido") from exc

    business_id = payload.get("business_id")
    if not isinstance(business_id, str) or not valid_business_id(business_id):
        raise AuthError("El token no incluye un business_id válido")
    return business_id


def mint_token(business_id: str, settings: Settings, hours: int | None = None) -> str:
    if not valid_business_id(business_id):
        raise ValueError("business_id inválido. Usa letras, números, _ o -, hasta 64 caracteres.")
    lifetime = settings.jwt_expire_hours if hours is None else hours
    if lifetime <= 0 or lifetime > 24 * 365:
        raise ValueError("La vigencia del token no es válida")
    now = datetime.now(timezone.utc)
    payload = {
        "business_id": business_id,
        "iat": now,
        "exp": now + timedelta(hours=lifetime),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    if isinstance(token, bytes):
        return token.decode("ascii")
    return token

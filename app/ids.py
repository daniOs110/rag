from __future__ import annotations

import re
import uuid

BUSINESS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CLIENT_ID_RE = re.compile(r"^cli_[0-9a-f]{32}$")
DOCUMENT_ID_RE = re.compile(r"^doc_[0-9a-f]{32}$")


def valid_business_id(value: str) -> bool:
    return bool(BUSINESS_ID_RE.fullmatch(value))


def valid_client_id(value: str) -> bool:
    return bool(CLIENT_ID_RE.fullmatch(value))


def new_client_id() -> str:
    return f"cli_{uuid.uuid4().hex}"


def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex}"

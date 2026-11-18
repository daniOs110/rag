from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import Settings
from app.container import wire
from app.errors import DocumentError, ProviderError
from app.routers import business, chat, clients, documents, health

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Client RAG API",
        version="0.1.0",
        summary="Clientes, documentos y chat RAG aislados por negocio.",
    )
    wire(app, settings)
    app.include_router(health.router)
    app.include_router(business.router)
    app.include_router(clients.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.exception_handler(DocumentError)
    async def document_error(_, exc: DocumentError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.detail})

    @app.exception_handler(ProviderError)
    async def provider_error(_, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": exc.detail})

    return app


def __getattr__(name: str):
    # Uvicorn carga app.main:app. Se crea al usarla para que importar create_app
    # en las pruebas no abra Chroma ni Firestore.
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

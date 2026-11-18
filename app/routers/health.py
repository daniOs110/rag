from fastapi import APIRouter, Request

from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    embeddings = request.app.state.embeddings
    chat = request.app.state.chat
    return HealthOut(
        status="ok",
        storage=request.app.state.storage_name,
        embeddings=embeddings.name,
        embedding_model=embeddings.model,
        llm=chat.name,
        llm_model=chat.model,
    )

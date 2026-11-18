from fastapi import APIRouter, Depends, Request

from app.deps import get_business_id, require_client
from app.schemas import ChatOut, ChatRequest, SourceOut

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatOut)
def chat(
    body: ChatRequest,
    request: Request,
    business_id: str = Depends(get_business_id),
) -> ChatOut:
    require_client(request, business_id, body.client_id)
    result = request.app.state.rag.answer(
        business_id=business_id,
        client_id=body.client_id,
        question=body.question,
        top_k=body.top_k,
    )
    return ChatOut(
        answer=result.answer,
        client_id=body.client_id,
        sources=[
            SourceOut(
                document_id=source.document_id,
                chunk_index=source.chunk_index,
                source=source.source,
                excerpt=source.excerpt,
                distance=source.distance,
            )
            for source in result.sources
        ],
    )

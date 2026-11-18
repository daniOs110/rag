from fastapi import APIRouter, Depends, Request

from app.deps import get_business_id
from app.schemas import BusinessOut, BusinessUpdate
from app.services.store import Business, utcnow

router = APIRouter(prefix="/business", tags=["business"])


@router.get("", response_model=BusinessOut)
def get_business(request: Request, business_id: str = Depends(get_business_id)) -> BusinessOut:
    stored = request.app.state.store.get_business(business_id)
    if stored is None or not stored.system_prompt:
        return BusinessOut(
            business_id=business_id,
            name=stored.name if stored and stored.name else business_id,
            system_prompt=request.app.state.settings.default_system_prompt,
            prompt_configured=False,
        )
    return BusinessOut(
        business_id=business_id,
        name=stored.name,
        system_prompt=stored.system_prompt,
        prompt_configured=True,
    )


@router.put("", response_model=BusinessOut)
def update_business(
    body: BusinessUpdate,
    request: Request,
    business_id: str = Depends(get_business_id),
) -> BusinessOut:
    stored = request.app.state.store.upsert_business(
        Business(
            business_id=business_id,
            name=body.name,
            system_prompt=body.system_prompt,
            updated_at=utcnow(),
        )
    )
    return BusinessOut(
        business_id=stored.business_id,
        name=stored.name,
        system_prompt=stored.system_prompt or body.system_prompt,
        prompt_configured=True,
    )

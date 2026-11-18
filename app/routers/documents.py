from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.deps import get_business_id, require_client
from app.schemas import UploadOut

router = APIRouter(prefix="/documents", tags=["documents"])

_TENANT_FIELDS = ("business_id", "clinic_id")


async def reject_tenant_in_body(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type and "application/x-www-form-urlencoded" not in content_type:
        return
    form = await request.form()
    if any(field in form for field in _TENANT_FIELDS):
        raise HTTPException(
            status_code=400,
            detail="business_id no se acepta en el cuerpo; sale del token",
        )


@router.post("/upload", response_model=UploadOut, status_code=201)
async def upload_document(
    request: Request,
    business_id: str = Depends(get_business_id),
    _: None = Depends(reject_tenant_in_body),
    file: UploadFile = File(...),
    client_id: str = Form(...),
) -> UploadOut:
    require_client(request, business_id, client_id)
    settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="El archivo supera el tamaño máximo")
    result = request.app.state.ingestion.ingest(
        data=data,
        filename=file.filename or "nota.txt",
        business_id=business_id,
        client_id=client_id,
    )
    return UploadOut(document_id=result.document_id, client_id=client_id, chunks=result.chunks)

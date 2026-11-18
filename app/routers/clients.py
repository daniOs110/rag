from fastapi import APIRouter, Depends, Query, Request

from app.deps import get_business_id
from app.ids import new_client_id
from app.schemas import ClientCreate, ClientOut
from app.services.store import Client, utcnow

router = APIRouter(prefix="/clients", tags=["clients"])


def _out(client: Client) -> ClientOut:
    return ClientOut(
        client_id=client.client_id,
        business_id=client.business_id,
        name=client.name,
        contact=client.contact,
        notes=client.notes,
        metadata=client.metadata,
        created_at=client.created_at,
    )


@router.post("", response_model=ClientOut, status_code=201)
def create_client(
    body: ClientCreate,
    request: Request,
    business_id: str = Depends(get_business_id),
) -> ClientOut:
    client = request.app.state.store.create_client(
        Client(
            client_id=new_client_id(),
            business_id=business_id,
            name=body.name,
            contact=body.contact,
            notes=body.notes,
            metadata=body.metadata,
            created_at=utcnow(),
        )
    )
    return _out(client)


@router.get("", response_model=list[ClientOut])
def list_clients(
    request: Request,
    business_id: str = Depends(get_business_id),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[ClientOut]:
    return [_out(client) for client in request.app.state.store.list_clients(business_id, limit)]

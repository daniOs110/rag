from app.auth import mint_token


def auth_header(app, business_id: str) -> dict[str, str]:
    token = mint_token(business_id, app.state.settings)
    return {"Authorization": f"Bearer {token}"}

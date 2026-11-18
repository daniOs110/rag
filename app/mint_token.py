from __future__ import annotations

import argparse
import sys

from app.auth import mint_token
from app.config import DEFAULT_JWT_SECRET, load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emite un JWT con el business_id del negocio.")
    parser.add_argument("--business-id", required=True)
    parser.add_argument("--hours", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    if settings.jwt_secret == DEFAULT_JWT_SECRET:
        print("Define JWT_SECRET antes de emitir tokens.", file=sys.stderr)
        raise SystemExit(1)
    try:
        token = mint_token(args.business_id, settings, hours=args.hours)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(token)


if __name__ == "__main__":
    main()

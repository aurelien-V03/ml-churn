"""API key check.

The key is read from `FAST_API_KEY` in the project's `.env`. With no key
configured the service refuses every prediction: an unusable service beats one
left open by accident.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Header carrying the key, the most widespread convention.
HEADER = "x-api-key"

# Paths left open: an orchestrator's probes do not authenticate, and the test
# page must load before it can ask for a key.
PUBLIC_PATHS = ("/health", "/ready", "/docs", "/redoc", "/openapi.json")


def expected_key() -> str | None:
    """Configured key, or `None` when the service has none."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("FAST_API_KEY") or None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Requires `X-API-Key` on the prediction endpoints."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method != "POST" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        expected = expected_key()
        if expected is None:
            return JSONResponse(
                {"detail": "FAST_API_KEY n'est pas configuree"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # `compare_digest` rather than `==`: comparison time must not depend on
        # how many characters are correct.
        provided = request.headers.get(HEADER, "")
        if not hmac.compare_digest(provided, expected):
            return JSONResponse(
                {"detail": f"cle d'API absente ou invalide, en-tete {HEADER}"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        return await call_next(request)

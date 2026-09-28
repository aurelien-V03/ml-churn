"""Request body size limit.

A plain ASGI middleware rather than `BaseHTTPMiddleware`: the body is counted
as it arrives, before FastAPI holds it in memory to parse it. A client that
announces an honest length then sends more is stopped too.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# 64 KB: a full gold row weighs under 4 KB, the margin is wide.
MAX_BODY_BYTES = 64 * 1024

# Drift is measured on a whole population, not on a single row: a hundred
# clients already weigh more than the default limit allows.
MAX_BODY_BYTES_BY_PATH: dict[str, int] = {"/drift": 1024 * 1024}

TOO_LARGE = 413


class BodySizeLimitMiddleware:
    """Refuses any request whose body exceeds `MAX_BODY_BYTES`."""

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int = MAX_BODY_BYTES,
        by_path: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.by_path = MAX_BODY_BYTES_BY_PATH if by_path is None else by_path

    def _limit(self, scope: Scope) -> int:
        """Limite applicable au chemin appele."""
        return self.by_path.get(scope.get("path", ""), self.max_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit(scope)

        # `content-length` lets us refuse without reading anything, when set.
        announced = Headers(scope=scope).get("content-length")
        if announced and announced.isdigit() and int(announced) > limit:
            await self._refuse(scope, receive, send, limit)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge:
            await self._refuse(scope, receive, send, limit)

    async def _refuse(
        self, scope: Scope, receive: Receive, send: Send, limit: int
    ) -> None:
        response = JSONResponse(
            {"detail": f"corps limite a {limit // 1024} ko"},
            status_code=TOO_LARGE,
        )
        await response(scope, receive, send)


class _BodyTooLarge(Exception):
    """Stops the read as soon as the limit is crossed."""

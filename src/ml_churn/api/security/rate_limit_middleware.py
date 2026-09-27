"""Per-client call quota.

Two budgets, counted separately: the prediction endpoints get a wide one, the
rest of the service a narrow one. A page polling the probes every few seconds
would otherwise eat the budget its predictions need.

Sliding window, held in the process memory: each instance counts on its own.
Behind several workers or replicas the effective limit is multiplied by their
number -- a shared counter (Redis) would be needed for a global one.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Calls allowed per client over the window, by budget.
LIMIT = 60
PREDICT_LIMIT = 400
WINDOW_S = 60

# Paths sharing the prediction budget.
PREDICT_PREFIX = "/predict"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Refuses beyond `LIMIT` calls per `WINDOW_S` seconds and per IP."""

    def __init__(
        self,
        app,
        limit: int = LIMIT,
        predict_limit: int = PREDICT_LIMIT,
        window_s: int = WINDOW_S,
    ) -> None:
        super().__init__(app)
        self.limit = limit
        self.predict_limit = predict_limit
        self.window_s = window_s
        self._calls: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def _client(self, request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _budget(self, request: Request) -> tuple[str, int]:
        """Budget applicable : celui des predictions, ou celui du reste."""
        if request.url.path.startswith(PREDICT_PREFIX):
            return "predict", self.predict_limit
        return "default", self.limit

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        now = time.monotonic()
        budget, limit = self._budget(request)
        calls = self._calls[self._client(request), budget]

        # The window slides: anything older than a minute is forgotten.
        while calls and now - calls[0] >= self.window_s:
            calls.popleft()

        if len(calls) >= limit:
            wait = self.window_s - (now - calls[0])
            return JSONResponse(
                {"detail": f"limite de {limit} appels par minute atteinte"},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"retry-after": str(max(1, round(wait)))},
            )

        calls.append(now)
        response = await call_next(request)
        response.headers["x-ratelimit-limit"] = str(limit)
        response.headers["x-ratelimit-remaining"] = str(limit - len(calls))
        return response

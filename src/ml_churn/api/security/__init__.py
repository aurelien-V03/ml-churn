"""Protections applied to incoming requests.

Three middlewares, in the order a request crosses them:

1. `BodySizeLimitMiddleware`: rejects oversized bodies, before any parsing.
2. `RateLimitMiddleware`: caps the number of calls per client and per minute.
3. `ApiKeyMiddleware`: requires the API key on the prediction endpoints.

Cheapest check first: no point validating a key on a request that is about to
be rejected for its size.
"""

from ml_churn.api.security.api_key_middleware import ApiKeyMiddleware
from ml_churn.api.security.body_size_middleware import BodySizeLimitMiddleware
from ml_churn.api.security.rate_limit_middleware import RateLimitMiddleware

__all__ = [
    "ApiKeyMiddleware",
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
]

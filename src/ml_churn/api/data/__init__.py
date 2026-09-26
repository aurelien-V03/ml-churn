"""Schemas des requetes et des reponses du service."""

from ml_churn.api.data.schemas import (
    HealthResponse,
    PredictChurnResponse,
    PredictClvResponse,
    PredictRequest,
    ReadyResponse,
)

__all__ = [
    "HealthResponse",
    "PredictChurnResponse",
    "PredictClvResponse",
    "PredictRequest",
    "ReadyResponse",
]

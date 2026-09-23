"""Schemas des requetes et des reponses du service."""

from ml_churn.api.data.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

__all__ = [
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "ReadyResponse",
]

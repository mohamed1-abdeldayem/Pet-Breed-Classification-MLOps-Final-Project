from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BreedScore(BaseModel):
    breed: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    breed: str
    species: Literal["cat", "dog"]
    confidence: float = Field(ge=0.0, le=1.0)
    top_3: list[BreedScore]
    decision: Literal["confident", "uncertain"]
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["healthy"]
    model_version: str
    num_classes: int
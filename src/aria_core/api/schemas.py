"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class CreateTenantRequest(BaseModel):
    slug: str = Field(min_length=2, max_length=63)
    name: str = Field(min_length=1, max_length=200)
    config: dict[str, Any] = Field(default_factory=dict)


class UpdateTenantConfigRequest(BaseModel):
    display_name: str | None = None
    logo_url: str | None = None
    default_model: str | None = None
    allowed_models: list[str] = Field(default_factory=list)
    max_tokens: int | None = None
    max_concurrent_agents: int = Field(default=10, ge=1, le=1000)
    max_plans_per_hour: int = Field(default=100, ge=1, le=10000)
    features: dict[str, bool] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreatePlanRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    actions: list[dict[str, Any]] = Field(default_factory=list)

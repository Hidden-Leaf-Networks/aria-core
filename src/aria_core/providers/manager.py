"""Provider manager — tenant-scoped LLM provider configuration and adapter creation.

Handles:
- API key storage per tenant per provider
- Model registry with capabilities and pricing
- Adapter factory: model name → correct adapter instance
- Provider health checks
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member


class ProviderType(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    GOOGLE = "google"
    LOCAL = "local"


class ModelInfo(BaseModel):
    """Model metadata in the registry."""

    id: str  # e.g. "gpt-4o", "claude-sonnet-4-20250514"
    name: str  # Human-readable name
    provider: ProviderType
    context_window: int = 128000
    max_output: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_extended_thinking: bool = False
    input_price_per_1m: float = 0.0  # $/1M input tokens
    output_price_per_1m: float = 0.0  # $/1M output tokens
    tags: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Per-tenant provider configuration."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = None
    provider: ProviderType
    api_key: str
    base_url: str | None = None
    default_model: str | None = None
    enabled: bool = True
    max_requests_per_minute: int = 60
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Built-in model registry
# ---------------------------------------------------------------------------

BUILTIN_MODELS: list[ModelInfo] = [
    # OpenAI
    ModelInfo(id="gpt-4o", name="GPT-4o", provider=ProviderType.OPENAI, context_window=128000, max_output=16384, supports_vision=True, input_price_per_1m=2.50, output_price_per_1m=10.00, tags=["flagship", "multimodal"]),
    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider=ProviderType.OPENAI, context_window=128000, max_output=16384, supports_vision=True, input_price_per_1m=0.15, output_price_per_1m=0.60, tags=["fast", "cheap"]),
    ModelInfo(id="gpt-4", name="GPT-4", provider=ProviderType.OPENAI, context_window=8192, max_output=4096, input_price_per_1m=30.00, output_price_per_1m=60.00, tags=["legacy"]),
    ModelInfo(id="o3", name="o3", provider=ProviderType.OPENAI, context_window=200000, max_output=100000, supports_extended_thinking=True, input_price_per_1m=10.00, output_price_per_1m=40.00, tags=["reasoning"]),
    ModelInfo(id="o4-mini", name="o4-mini", provider=ProviderType.OPENAI, context_window=200000, max_output=100000, supports_extended_thinking=True, input_price_per_1m=1.10, output_price_per_1m=4.40, tags=["reasoning", "fast"]),

    # Anthropic
    ModelInfo(id="claude-opus-4-20250514", name="Claude Opus 4", provider=ProviderType.ANTHROPIC, context_window=200000, max_output=32000, supports_vision=True, supports_extended_thinking=True, input_price_per_1m=15.00, output_price_per_1m=75.00, tags=["flagship", "reasoning"]),
    ModelInfo(id="claude-sonnet-4-20250514", name="Claude Sonnet 4", provider=ProviderType.ANTHROPIC, context_window=200000, max_output=16000, supports_vision=True, supports_extended_thinking=True, input_price_per_1m=3.00, output_price_per_1m=15.00, tags=["balanced"]),
    ModelInfo(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5", provider=ProviderType.ANTHROPIC, context_window=200000, max_output=8192, supports_vision=True, input_price_per_1m=0.80, output_price_per_1m=4.00, tags=["fast", "cheap"]),

    # xAI
    ModelInfo(id="grok-2-latest", name="Grok 2", provider=ProviderType.XAI, context_window=131072, max_output=8192, supports_vision=True, input_price_per_1m=2.00, output_price_per_1m=10.00, tags=["balanced"]),
    ModelInfo(id="grok-3-mini", name="Grok 3 Mini", provider=ProviderType.XAI, context_window=131072, max_output=8192, supports_extended_thinking=True, input_price_per_1m=0.30, output_price_per_1m=0.50, tags=["reasoning", "fast"]),

    # Google
    ModelInfo(id="gemini-2.5-pro", name="Gemini 2.5 Pro", provider=ProviderType.GOOGLE, context_window=1000000, max_output=65536, supports_vision=True, supports_extended_thinking=True, input_price_per_1m=1.25, output_price_per_1m=10.00, tags=["flagship", "long-context"]),
    ModelInfo(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider=ProviderType.GOOGLE, context_window=1000000, max_output=65536, supports_vision=True, input_price_per_1m=0.15, output_price_per_1m=0.60, tags=["fast", "cheap", "long-context"]),
]


class ProviderManager:
    """Manages LLM provider configurations per tenant and creates adapters."""

    def __init__(self) -> None:
        # {tenant_id: {provider_type: ProviderConfig}}
        self._configs: dict[UUID, dict[ProviderType, ProviderConfig]] = {}
        self._models: dict[str, ModelInfo] = {m.id: m for m in BUILTIN_MODELS}

    # -------------------------------------------------------------------
    # Provider configuration
    # -------------------------------------------------------------------

    def configure(self, tenant_id: UUID, config: ProviderConfig) -> ProviderConfig:
        """Register or update a provider config for a tenant."""
        config = config.model_copy(update={"tenant_id": tenant_id})
        if tenant_id not in self._configs:
            self._configs[tenant_id] = {}
        self._configs[tenant_id][config.provider] = config
        return config

    def get_config(self, tenant_id: UUID, provider: ProviderType) -> ProviderConfig | None:
        return self._configs.get(tenant_id, {}).get(provider)

    def list_configs(self, tenant_id: UUID) -> list[ProviderConfig]:
        return list(self._configs.get(tenant_id, {}).values())

    def remove_config(self, tenant_id: UUID, provider: ProviderType) -> bool:
        configs = self._configs.get(tenant_id, {})
        if provider in configs:
            del configs[provider]
            return True
        return False

    def get_configured_providers(self, tenant_id: UUID) -> list[ProviderType]:
        """List providers that have API keys configured for a tenant."""
        return [
            config.provider
            for config in self._configs.get(tenant_id, {}).values()
            if config.enabled and config.api_key
        ]

    # -------------------------------------------------------------------
    # Model registry
    # -------------------------------------------------------------------

    def list_models(
        self,
        provider: ProviderType | None = None,
        tag: str | None = None,
        tenant_id: UUID | None = None,
    ) -> list[ModelInfo]:
        """List available models, optionally filtered."""
        models = list(self._models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if tag:
            models = [m for m in models if tag in m.tags]

        if tenant_id:
            configured = set(self.get_configured_providers(tenant_id))
            models = [m for m in models if m.provider in configured]

        return sorted(models, key=lambda m: m.id)

    def get_model(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def register_model(self, model: ModelInfo) -> None:
        """Register a custom model."""
        self._models[model.id] = model

    # -------------------------------------------------------------------
    # Adapter factory
    # -------------------------------------------------------------------

    def get_adapter(self, tenant_id: UUID, model_id: str) -> Any:
        """Create the correct adapter for a model, using the tenant's API key.

        Returns a ModelAdapter instance ready to use.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Unknown model: {model_id}")

        config = self.get_config(tenant_id, model.provider)
        if not config:
            raise ValueError(
                f"Provider '{model.provider}' not configured for tenant {tenant_id}. "
                f"Call configure() with an API key first."
            )

        if not config.enabled:
            raise ValueError(f"Provider '{model.provider}' is disabled for tenant {tenant_id}")

        if model.provider == ProviderType.OPENAI:
            from aria_core.adapters.openai import OpenAIAdapter
            return OpenAIAdapter(
                api_key=config.api_key,
                model=model_id,
                base_url=config.base_url,
            )

        elif model.provider == ProviderType.ANTHROPIC:
            from aria_core.adapters.anthropic import AnthropicAdapter
            return AnthropicAdapter(
                api_key=config.api_key,
                model=model_id,
            )

        elif model.provider == ProviderType.XAI:
            from aria_core.adapters.xai import XAIAdapter
            return XAIAdapter(
                api_key=config.api_key,
                model=model_id,
            )

        elif model.provider == ProviderType.GOOGLE:
            # Google uses OpenAI-compatible API via AI Gateway
            from aria_core.adapters.openai import OpenAIAdapter
            return OpenAIAdapter(
                api_key=config.api_key,
                model=model_id,
                base_url=config.base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
            )

        elif model.provider == ProviderType.LOCAL:
            from aria_core.adapters.openai import OpenAIAdapter
            return OpenAIAdapter(
                api_key=config.api_key or "local",
                model=model_id,
                base_url=config.base_url or "http://localhost:11434/v1",
            )

        else:
            raise ValueError(f"Unsupported provider: {model.provider}")

    def get_best_adapter(self, tenant_id: UUID, preference: str = "balanced") -> Any:
        """Get the best available adapter based on preference.

        Preferences: "fast", "cheap", "flagship", "reasoning", "balanced"
        Falls back to any available model.
        """
        configured = self.get_configured_providers(tenant_id)
        if not configured:
            raise ValueError(f"No providers configured for tenant {tenant_id}")

        available = self.list_models(tenant_id=tenant_id)
        if not available:
            raise ValueError("No models available for configured providers")

        # Filter by preference tag
        preferred = [m for m in available if preference in m.tags]
        if preferred:
            return self.get_adapter(tenant_id, preferred[0].id)

        # Fallback: first available
        return self.get_adapter(tenant_id, available[0].id)

    # -------------------------------------------------------------------
    # Health / status
    # -------------------------------------------------------------------

    def get_status(self, tenant_id: UUID) -> dict[str, Any]:
        """Get provider status for a tenant."""
        configs = self.list_configs(tenant_id)
        configured_providers = self.get_configured_providers(tenant_id)
        available_models = self.list_models(tenant_id=tenant_id)

        return {
            "tenant_id": str(tenant_id),
            "providers": [
                {
                    "provider": c.provider.value,
                    "enabled": c.enabled,
                    "default_model": c.default_model,
                    "has_key": bool(c.api_key),
                }
                for c in configs
            ],
            "configured_count": len(configured_providers),
            "available_models": len(available_models),
            "model_ids": [m.id for m in available_models],
        }

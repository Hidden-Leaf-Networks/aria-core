"""Model provider management — configure, register, and route to LLM providers.

Provides:
- ProviderConfig: per-tenant API key + model settings
- ModelRegistry: available models across all providers
- AdapterFactory: create the right adapter from model name
- ProviderManager: tenant-scoped provider CRUD

Usage:
    from aria_core.providers import ProviderManager

    pm = ProviderManager()
    pm.configure(tenant_id, ProviderConfig(provider="openai", api_key="sk-..."))
    adapter = pm.get_adapter(tenant_id, "gpt-4o")
    response = await adapter.generate_response(context)
"""

from aria_core.providers.manager import ProviderManager, ProviderConfig, ModelInfo

__all__ = ["ModelInfo", "ProviderConfig", "ProviderManager"]

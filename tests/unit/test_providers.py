"""Tests for provider management."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.providers.manager import (
    BUILTIN_MODELS,
    ModelInfo,
    ProviderConfig,
    ProviderManager,
    ProviderType,
)


class TestModelRegistry:
    def test_builtin_models_loaded(self) -> None:
        pm = ProviderManager()
        assert len(pm.list_models()) == len(BUILTIN_MODELS)

    def test_list_by_provider(self) -> None:
        pm = ProviderManager()
        openai_models = pm.list_models(provider=ProviderType.OPENAI)
        assert all(m.provider == ProviderType.OPENAI for m in openai_models)
        assert len(openai_models) >= 3

    def test_list_by_tag(self) -> None:
        pm = ProviderManager()
        flagships = pm.list_models(tag="flagship")
        assert all("flagship" in m.tags for m in flagships)

    def test_get_model(self) -> None:
        pm = ProviderManager()
        model = pm.get_model("gpt-4o")
        assert model is not None
        assert model.name == "GPT-4o"
        assert model.provider == ProviderType.OPENAI

    def test_register_custom_model(self) -> None:
        pm = ProviderManager()
        custom = ModelInfo(
            id="local-llama", name="Local Llama", provider=ProviderType.LOCAL,
            context_window=8192, input_price_per_1m=0, output_price_per_1m=0,
        )
        pm.register_model(custom)
        assert pm.get_model("local-llama") is not None


class TestProviderConfig:
    def test_configure_provider(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        config = pm.configure(tid, ProviderConfig(
            provider=ProviderType.OPENAI, api_key="sk-test123",
        ))
        assert config.tenant_id == tid
        assert config.api_key == "sk-test123"

    def test_list_configs(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-1"))
        pm.configure(tid, ProviderConfig(provider=ProviderType.ANTHROPIC, api_key="sk-2"))
        assert len(pm.list_configs(tid)) == 2

    def test_remove_config(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-1"))
        assert pm.remove_config(tid, ProviderType.OPENAI) is True
        assert pm.remove_config(tid, ProviderType.OPENAI) is False

    def test_configured_providers(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-1"))
        pm.configure(tid, ProviderConfig(provider=ProviderType.XAI, api_key="xai-1"))
        providers = pm.get_configured_providers(tid)
        assert ProviderType.OPENAI in providers
        assert ProviderType.XAI in providers
        assert ProviderType.ANTHROPIC not in providers

    def test_disabled_provider_excluded(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-1", enabled=False))
        assert len(pm.get_configured_providers(tid)) == 0


class TestAdapterFactory:
    def test_get_adapter_openai(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-test"))
        adapter = pm.get_adapter(tid, "gpt-4o")
        assert adapter is not None

    def test_get_adapter_anthropic(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.ANTHROPIC, api_key="sk-ant-test"))
        adapter = pm.get_adapter(tid, "claude-sonnet-4-20250514")
        assert adapter is not None

    def test_get_adapter_xai(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.XAI, api_key="xai-test"))
        adapter = pm.get_adapter(tid, "grok-2-latest")
        assert adapter is not None

    def test_unknown_model_raises(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        with pytest.raises(ValueError, match="Unknown model"):
            pm.get_adapter(tid, "nonexistent-model")

    def test_unconfigured_provider_raises(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        with pytest.raises(ValueError, match="not configured"):
            pm.get_adapter(tid, "gpt-4o")

    def test_get_best_adapter(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-test"))
        adapter = pm.get_best_adapter(tid, "fast")
        assert adapter is not None

    def test_no_providers_raises(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        with pytest.raises(ValueError, match="No providers configured"):
            pm.get_best_adapter(tid)


class TestAvailableModels:
    def test_list_available_for_tenant(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.ANTHROPIC, api_key="sk-ant"))
        available = pm.list_models(tenant_id=tid)
        assert all(m.provider == ProviderType.ANTHROPIC for m in available)

    def test_no_config_no_available(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        available = pm.list_models(tenant_id=tid)
        assert len(available) == 0


class TestProviderStatus:
    def test_status_report(self) -> None:
        pm = ProviderManager()
        tid = uuid4()
        pm.configure(tid, ProviderConfig(provider=ProviderType.OPENAI, api_key="sk-1"))
        pm.configure(tid, ProviderConfig(provider=ProviderType.ANTHROPIC, api_key="sk-2"))

        status = pm.get_status(tid)
        assert status["configured_count"] == 2
        assert status["available_models"] > 0
        assert len(status["providers"]) == 2

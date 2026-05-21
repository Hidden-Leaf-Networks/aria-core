"""Tests for ARIA-296: GitHub one-click deploy."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.deploy import (
    AriaYamlConfig,
    DeployConfig,
    DeploymentRecord,
    GitHubDeployManager,
)

SAMPLE_YAML = """\
agents:
  - name: Research Bot
    slug: research-bot
    model: claude-sonnet-4-20250514
    system_prompt: "You are a research analyst..."
    max_steps: 20
  - name: Writer Bot
    slug: writer-bot
    model: gpt-4
    system_prompt: "You write content."

plans:
  - name: Daily Report
    actions:
      - name: Gather Data
        skill_name: web_search
      - name: Analyze
        skill_name: synthesize
        dependencies: [0]

settings:
  default_model: gpt-4
  risk_threshold: 50
  features:
    streaming: true
"""


@pytest.fixture
def manager() -> GitHubDeployManager:
    return GitHubDeployManager()


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.fixture
def sample_config(tenant_id):
    return DeployConfig(
        repo_url="https://github.com/org/repo.git",
        branch="main",
        tenant_id=tenant_id,
        auto_deploy=True,
    )


class TestDeployConfig:
    def test_defaults(self, tenant_id) -> None:
        config = DeployConfig(
            repo_url="https://github.com/org/repo.git",
            tenant_id=tenant_id,
        )
        assert config.branch == "main"
        assert config.config_path == "aria.yaml"
        assert config.auto_deploy is True
        assert config.preview_on_pr is False
        assert config.webhook_secret is None
        assert config.last_deployed_at is None

    def test_custom_values(self, tenant_id) -> None:
        config = DeployConfig(
            repo_url="https://github.com/org/repo.git",
            branch="develop",
            config_path=".aria/config.yaml",
            tenant_id=tenant_id,
            auto_deploy=False,
            preview_on_pr=True,
            webhook_secret="secret123",
        )
        assert config.branch == "develop"
        assert config.config_path == ".aria/config.yaml"
        assert config.auto_deploy is False
        assert config.preview_on_pr is True


class TestRepoRegistration:
    def test_register_repo(self, manager, sample_config) -> None:
        result = manager.register_repo(sample_config)
        assert result.repo_url == sample_config.repo_url
        assert result.tenant_id == sample_config.tenant_id

    def test_list_repos(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        repos = manager.list_repos(tenant_id)
        assert len(repos) == 1
        assert repos[0].repo_url == sample_config.repo_url

    def test_list_repos_tenant_isolation(self, manager, sample_config) -> None:
        manager.register_repo(sample_config)
        other_tenant = uuid4()
        repos = manager.list_repos(other_tenant)
        assert len(repos) == 0

    def test_unregister_repo(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        result = manager.unregister_repo(sample_config.repo_url, tenant_id)
        assert result is True
        assert manager.list_repos(tenant_id) == []

    def test_unregister_nonexistent(self, manager, tenant_id) -> None:
        result = manager.unregister_repo("https://github.com/x/y.git", tenant_id)
        assert result is False


class TestDeployment:
    def test_deploy_success(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        record = manager.deploy(sample_config.repo_url, tenant_id)
        assert record.status == "success"
        assert record.agents_deployed >= 1
        assert record.plans_deployed >= 1
        assert record.completed_at is not None
        assert record.error is None

    def test_deploy_unregistered_repo_fails(self, manager, tenant_id) -> None:
        record = manager.deploy("https://github.com/x/y.git", tenant_id)
        assert record.status == "failed"
        assert "not registered" in record.error

    def test_deploy_updates_last_deployed(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        manager.deploy(sample_config.repo_url, tenant_id)
        repos = manager.list_repos(tenant_id)
        assert repos[0].last_deployed_at is not None

    def test_get_deployment(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        record = manager.deploy(sample_config.repo_url, tenant_id)
        fetched = manager.get_deployment(record.id)
        assert fetched.id == record.id

    def test_get_deployment_not_found(self, manager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.get_deployment(uuid4())

    def test_list_deployments(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        manager.deploy(sample_config.repo_url, tenant_id, commit_sha="abc123")
        manager.deploy(sample_config.repo_url, tenant_id, commit_sha="def456")
        records = manager.list_deployments(tenant_id)
        assert len(records) == 2

    def test_list_deployments_limit(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        for i in range(5):
            manager.deploy(sample_config.repo_url, tenant_id, commit_sha=f"sha{i}")
        records = manager.list_deployments(tenant_id, limit=3)
        assert len(records) == 3


class TestRollback:
    def test_rollback_success(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        record = manager.deploy(sample_config.repo_url, tenant_id)
        rolled = manager.rollback(record.id)
        assert rolled.status == "rolled_back"

    def test_rollback_nonexistent(self, manager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.rollback(uuid4())

    def test_rollback_failed_deployment(self, manager, tenant_id) -> None:
        record = manager.deploy("https://github.com/x/y.git", tenant_id)
        assert record.status == "failed"
        with pytest.raises(ValueError, match="Cannot rollback"):
            manager.rollback(record.id)


class TestWebhook:
    def test_handle_push_webhook(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        payload = {
            "ref": "refs/heads/main",
            "after": "abc123def",
            "repository": {
                "clone_url": sample_config.repo_url,
            },
        }
        record = manager.handle_webhook(payload)
        assert record is not None
        assert record.status == "success"
        assert record.commit_sha == "abc123def"

    def test_handle_webhook_wrong_branch(self, manager, sample_config, tenant_id) -> None:
        manager.register_repo(sample_config)
        payload = {
            "ref": "refs/heads/develop",
            "after": "abc123def",
            "repository": {"clone_url": sample_config.repo_url},
        }
        result = manager.handle_webhook(payload)
        assert result is None

    def test_handle_webhook_no_auto_deploy(self, manager, tenant_id) -> None:
        config = DeployConfig(
            repo_url="https://github.com/org/repo.git",
            tenant_id=tenant_id,
            auto_deploy=False,
        )
        manager.register_repo(config)
        payload = {
            "ref": "refs/heads/main",
            "after": "sha123",
            "repository": {"clone_url": config.repo_url},
        }
        result = manager.handle_webhook(payload)
        assert result is None

    def test_handle_webhook_invalid_payload(self, manager) -> None:
        result = manager.handle_webhook({})
        assert result is None


class TestYamlParsing:
    def test_parse_full_yaml(self, manager) -> None:
        config = manager.parse_aria_yaml(SAMPLE_YAML)
        assert isinstance(config, AriaYamlConfig)
        assert len(config.agents) == 2
        assert config.agents[0].name == "Research Bot"
        assert config.agents[0].slug == "research-bot"
        assert config.agents[0].max_steps == 20
        assert len(config.plans) == 1
        assert config.plans[0].name == "Daily Report"
        assert len(config.plans[0].actions) == 2
        assert config.plans[0].actions[1].dependencies == [0]
        assert config.settings["default_model"] == "gpt-4"
        assert config.settings["risk_threshold"] == 50

    def test_parse_empty_yaml(self, manager) -> None:
        config = manager.parse_aria_yaml("---\n{}")
        assert config.agents == []
        assert config.plans == []
        assert config.settings == {}

    def test_parse_invalid_yaml_raises(self, manager) -> None:
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            manager.parse_aria_yaml("- just a list")

    def test_parse_agents_only(self, manager) -> None:
        content = """\
agents:
  - name: Solo Agent
    slug: solo
    model: gpt-4
"""
        config = manager.parse_aria_yaml(content)
        assert len(config.agents) == 1
        assert config.agents[0].slug == "solo"
        assert config.plans == []

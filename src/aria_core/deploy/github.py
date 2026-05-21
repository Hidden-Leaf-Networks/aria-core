"""GitHub-based deployment manager for ARIA agent configurations.

Allows users to connect a GitHub repository containing an aria.yaml config
and deploy agent configurations with a single action.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DeployConfig(BaseModel):
    """Configuration for a connected GitHub repository."""

    repo_url: str
    branch: str = "main"
    config_path: str = "aria.yaml"
    tenant_id: UUID
    auto_deploy: bool = True
    preview_on_pr: bool = False
    webhook_secret: str | None = None
    last_deployed_at: datetime | None = None


class AgentSpec(BaseModel):
    """Agent specification inside aria.yaml."""

    name: str
    slug: str
    model: str = "gpt-4"
    system_prompt: str | None = None
    max_steps: int = 10


class PlanAction(BaseModel):
    """A single action within a plan template."""

    name: str
    skill_name: str
    dependencies: list[int] = Field(default_factory=list)


class PlanSpec(BaseModel):
    """Plan template specification inside aria.yaml."""

    name: str
    actions: list[PlanAction] = Field(default_factory=list)


class AriaYamlConfig(BaseModel):
    """Parsed representation of a repo's aria.yaml file."""

    agents: list[AgentSpec] = Field(default_factory=list)
    plans: list[PlanSpec] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class DeploymentRecord(BaseModel):
    """Record of a single deployment attempt."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    repo_url: str
    branch: str
    commit_sha: str
    status: Literal["pending", "deploying", "success", "failed", "rolled_back"] = (
        "pending"
    )
    agents_deployed: int = 0
    plans_deployed: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class GitHubDeployManager:
    """Manages GitHub repository connections and deployments."""

    def __init__(self) -> None:
        self._repos: dict[str, DeployConfig] = {}  # key: f"{tenant_id}:{repo_url}"
        self._deployments: dict[UUID, DeploymentRecord] = {}

    def _repo_key(self, tenant_id: UUID, repo_url: str) -> str:
        return f"{tenant_id}:{repo_url}"

    # --- Repo management ---

    def register_repo(self, config: DeployConfig) -> DeployConfig:
        """Register a GitHub repository for deployment."""
        key = self._repo_key(config.tenant_id, config.repo_url)
        self._repos[key] = config
        return config

    def list_repos(self, tenant_id: UUID) -> list[DeployConfig]:
        """List all registered repos for a tenant."""
        return [c for c in self._repos.values() if c.tenant_id == tenant_id]

    def unregister_repo(self, repo_url: str, tenant_id: UUID) -> bool:
        """Remove a registered repository. Returns True if found and removed."""
        key = self._repo_key(tenant_id, repo_url)
        if key in self._repos:
            del self._repos[key]
            return True
        return False

    # --- Deployment ---

    def deploy(
        self,
        repo_url: str,
        tenant_id: UUID,
        commit_sha: str = "HEAD",
    ) -> DeploymentRecord:
        """Simulate deploying the latest config from a connected repo."""
        key = self._repo_key(tenant_id, repo_url)
        config = self._repos.get(key)
        if config is None:
            record = DeploymentRecord(
                tenant_id=tenant_id,
                repo_url=repo_url,
                branch="main",
                commit_sha=commit_sha,
                status="failed",
                error="Repository not registered",
            )
            self._deployments[record.id] = record
            return record

        record = DeploymentRecord(
            tenant_id=tenant_id,
            repo_url=repo_url,
            branch=config.branch,
            commit_sha=commit_sha,
            status="deploying",
        )

        # Simulate successful deployment
        record.status = "success"
        record.agents_deployed = 1
        record.plans_deployed = 1
        record.completed_at = datetime.now(timezone.utc)

        # Update last deployed
        config.last_deployed_at = record.completed_at

        self._deployments[record.id] = record
        return record

    def rollback(self, deployment_id: UUID) -> DeploymentRecord:
        """Roll back a deployment by ID."""
        record = self._deployments.get(deployment_id)
        if record is None:
            raise ValueError(f"Deployment {deployment_id} not found")

        if record.status != "success":
            raise ValueError(
                f"Cannot rollback deployment with status '{record.status}'"
            )

        record.status = "rolled_back"
        record.completed_at = datetime.now(timezone.utc)
        return record

    def get_deployment(self, deployment_id: UUID) -> DeploymentRecord:
        """Retrieve a deployment record by ID."""
        record = self._deployments.get(deployment_id)
        if record is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        return record

    def list_deployments(
        self, tenant_id: UUID, limit: int = 20
    ) -> list[DeploymentRecord]:
        """List deployments for a tenant, most recent first."""
        records = [
            r for r in self._deployments.values() if r.tenant_id == tenant_id
        ]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    # --- Webhook ---

    def handle_webhook(self, payload: dict[str, Any]) -> DeploymentRecord | None:
        """Handle a GitHub push webhook payload.

        Expected payload keys:
        - repository.clone_url or repository.html_url
        - ref (e.g. "refs/heads/main")
        - after (commit SHA)
        """
        repo_info = payload.get("repository", {})
        repo_url = repo_info.get("clone_url") or repo_info.get("html_url", "")
        ref = payload.get("ref", "")
        commit_sha = payload.get("after", "HEAD")

        # Extract branch from ref
        branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ""

        if not repo_url or not branch:
            return None

        # Find matching config
        for config in self._repos.values():
            if config.repo_url == repo_url and config.branch == branch:
                if config.auto_deploy:
                    return self.deploy(repo_url, config.tenant_id, commit_sha)
        return None

    # --- YAML parsing ---

    def parse_aria_yaml(self, content: str) -> AriaYamlConfig:
        """Parse an aria.yaml file content into an AriaYamlConfig model."""
        if not YAML_AVAILABLE:
            raise RuntimeError(
                "PyYAML is not installed. Install it with: pip install pyyaml"
            )

        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("aria.yaml must be a YAML mapping at the top level")

        return AriaYamlConfig(
            agents=[AgentSpec(**a) for a in data.get("agents", [])],
            plans=[
                PlanSpec(
                    name=p["name"],
                    actions=[PlanAction(**act) for act in p.get("actions", [])],
                )
                for p in data.get("plans", [])
            ],
            settings=data.get("settings", {}),
        )

"""Fine-grained capability-based auth for Aria Core.

Extends the existing JWT RBAC (admin/operator/viewer) with per-skill,
per-resource permission grants via capability tokens.

ARIA-305: Fine-grained Auth
"""

from aria_core.auth.capabilities import (
    Capability,
    CapabilityChecker,
    CapabilityToken,
    ResourceType,
    SkillPermissionGuard,
)

__all__ = [
    "Capability",
    "CapabilityChecker",
    "CapabilityToken",
    "ResourceType",
    "SkillPermissionGuard",
]

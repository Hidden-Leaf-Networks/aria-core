"""Tenant system — multi-tenant isolation for white-label deployments.

Provides tenant configuration, context threading, and isolation primitives.

Usage:
    from aria_core.tenant import Tenant, TenantConfig, TenantContext
"""

from aria_core.tenant.models import Tenant, TenantConfig, TenantContext
from aria_core.tenant.config_resolver import ConfigResolver

__all__ = [
    "ConfigResolver",
    "Tenant",
    "TenantConfig",
    "TenantContext",
]

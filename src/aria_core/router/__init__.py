"""Intent-aware routing with pluggable strategies."""

from aria_core.router.router import Router, create_default_router
from aria_core.router.strategies import (
    ClarifyStrategy,
    DirectStrategy,
    PlanStrategy,
    RouteResult,
    RoutingStrategy,
)

__all__ = [
    "ClarifyStrategy",
    "DirectStrategy",
    "PlanStrategy",
    "RouteResult",
    "Router",
    "RoutingStrategy",
    "create_default_router",
]

"""
Aria Core — Deterministic AI Agent Framework

A production-grade framework for building AI agents with:
- FSM-based deterministic execution
- Multi-model orchestration (Deep Bridge consensus)
- Intent-aware routing with pluggable strategies
- Permission-first safety with risk scoring
- Plan engine with dependency tracking and audit trails

Built by Hidden Leaf Networks.
https://github.com/Hidden-Leaf-Networks/aria-core
"""

__version__ = "0.1.0"

# Runtime
from aria_core.runtime.models import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentStateEnum,
    ChatMessage,
    MessageRole,
)
from aria_core.runtime.state_machine import AgentStateMachine
from aria_core.runtime.states import State, STATE_REGISTRY
from aria_core.runtime.transitions import Transition

# Orchestration
from aria_core.orchestration.deep_bridge import (
    ConsensusMode,
    DeepBridgeValidator,
)

# Router
from aria_core.router.router import Router
from aria_core.router.strategies import RoutingStrategy, RouteResult

# Permissions
from aria_core.permissions.risk_engine import RiskEngine
from aria_core.permissions.models import (
    SkillCategory,
    ImpactScope,
    RiskScoreInput,
)

# Planning
from aria_core.planning.plan_engine import PlanEngine

# Adapters
from aria_core.adapters import (
    AnthropicAdapter,
    ModelAdapter,
    OpenAIAdapter,
    OpenAIAdapterStub,
    XAIAdapter,
)

__all__ = [
    # Runtime
    "AgentConfig",
    "AgentContext",
    "AgentResult",
    "AgentStateMachine",
    "AgentStateEnum",
    "ChatMessage",
    "MessageRole",
    "State",
    "STATE_REGISTRY",
    "Transition",
    # Orchestration
    "ConsensusMode",
    "DeepBridgeValidator",
    # Router
    "RouteResult",
    "Router",
    "RoutingStrategy",
    # Permissions
    "ImpactScope",
    "RiskEngine",
    "RiskScoreInput",
    "SkillCategory",
    # Planning
    "PlanEngine",
    # Adapters
    "AnthropicAdapter",
    "ModelAdapter",
    "OpenAIAdapter",
    "OpenAIAdapterStub",
    "XAIAdapter",
]

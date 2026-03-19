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

__all__ = [
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
]

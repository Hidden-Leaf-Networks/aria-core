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

from aria_core.runtime.state_machine import AgentStateMachine, AgentState
from aria_core.router.router import Router
from aria_core.orchestration.deep_bridge import DeepBridgeValidator
from aria_core.permissions.risk_engine import RiskEngine
from aria_core.planning.plan_engine import PlanEngine

__all__ = [
    "AgentStateMachine",
    "AgentState",
    "Router",
    "DeepBridgeValidator",
    "RiskEngine",
    "PlanEngine",
]

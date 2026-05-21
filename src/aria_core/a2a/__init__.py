"""A2A (Agent-to-Agent) protocol integration for Aria Core.

Implements the Google A2A protocol for cross-vendor agent communication.
Governed by the Agentic AI Foundation (AAIF) under the Linux Foundation.

Provides:
- AgentCard: Discovery document for aria-core agents
- A2AServer: Handles incoming A2A task requests
- A2AClient: Delegates tasks to external A2A agents

Protocol: JSON-RPC 2.0 over HTTP(S)
Discovery: GET /.well-known/a2a/agent-card
Spec: https://a2a-protocol.org/latest/specification/
"""

from aria_core.a2a.models import AgentCard, A2ATask, TaskState, A2AMessage, A2APart
from aria_core.a2a.server import A2AServer
from aria_core.a2a.client import A2AClient

__all__ = [
    "AgentCard",
    "A2AClient",
    "A2AMessage",
    "A2APart",
    "A2AServer",
    "A2ATask",
    "TaskState",
]

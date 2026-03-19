# Aria Core

**Deterministic AI agent framework with multi-model orchestration, FSM execution, and permission-first safety.**

Built by [Hidden Leaf Networks](https://hiddenleafnetworks.com).

> Aria Core is the open-source foundation extracted from [Aria](https://hiddenleafnetworks.com/portfolio) — a production agent system running 4 autonomous agents across DevRel, research, trading, and content domains.

---

## What is Aria Core?

Aria Core is a Python framework for building AI agents that are **deterministic, safe, and observable**. Unlike prompt-chain frameworks that hope for the best, Aria Core guarantees:

- **No uncontrolled loops** — FSM state machine with validated transitions and max-step enforcement
- **Multi-model consensus** — Deep Bridge queries multiple LLMs in parallel for high-stakes decisions
- **Permission-first safety** — every tool execution goes through risk scoring and approval gates
- **Full audit trail** — every state transition, tool call, and decision is logged and inspectable
- **Intent-aware routing** — requests are classified and routed to the right execution strategy

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Your Agent                     │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Router          Intent classification &         │
│                  strategy selection               │
├─────────────────────────────────────────────────┤
│  Planner         Plan generation with            │
│                  dependency tracking              │
├─────────────────────────────────────────────────┤
│  Runtime (FSM)   Deterministic state machine     │
│                  IDLE → ROUTING → PLANNING →     │
│                  EXECUTING → RESPONDING → DONE   │
├─────────────────────────────────────────────────┤
│  Permissions     Risk scoring, approval gates,   │
│                  audit trail                      │
├─────────────────────────────────────────────────┤
│  Deep Bridge     Multi-model consensus for       │
│                  high-stakes decisions            │
├─────────────────────────────────────────────────┤
│  Adapters        OpenAI │ Anthropic │ xAI        │
└─────────────────────────────────────────────────┘
```

## Installation

```bash
pip install aria-core

# With LLM providers
pip install aria-core[openai]          # OpenAI only
pip install aria-core[anthropic]       # Anthropic only
pip install aria-core[all-providers]   # All providers
```

## Quick Start

```python
from aria_core import AgentStateMachine, Router, DeepBridgeValidator, RiskEngine

# Create a basic agent with deterministic execution
agent = AgentStateMachine(max_steps=10, timeout=300)

# Route incoming requests by intent
router = Router(strategies=[...])

# Add multi-model consensus for high-risk actions
bridge = DeepBridgeValidator(
    providers=["openai", "anthropic"],
    consensus_mode="MAJORITY",
)

# Score risk and enforce approval gates
risk = RiskEngine(threshold=0.7)
```

> Full documentation and examples coming soon.

## Modules

| Module | Description |
|--------|-------------|
| `aria_core.runtime` | FSM-based state machine with deterministic execution guarantees |
| `aria_core.router` | Intent classification and strategy-based routing |
| `aria_core.orchestration` | Deep Bridge multi-model consensus validation |
| `aria_core.planning` | Plan lifecycle, action dependencies, versioning |
| `aria_core.permissions` | Risk scoring, approval workflows, audit trails |
| `aria_core.adapters` | LLM provider adapters (OpenAI, Anthropic, xAI) |

## Production Heritage

Aria Core isn't theoretical — it's extracted from a production system:

- **4 autonomous agents** running in production (DevRel, research, trading, content)
- **18 governed skills** across SAFE/LOW/MEDIUM/HIGH risk tiers
- **Multi-model orchestration** across OpenAI, Anthropic, and xAI
- **Full audit trail** with PostgreSQL event store
- **Real-time SSE streaming** for observability

## Why Aria Core?

| Feature | Aria Core | LangChain | CrewAI |
|---------|-----------|-----------|--------|
| Deterministic execution | FSM with validated transitions | No guarantee | No guarantee |
| Multi-model consensus | Deep Bridge voting | Basic routing | None |
| Permission system | Risk scoring + approval gates | None | None |
| Audit trail | Full event store | Callbacks only | None |
| State management | Explicit FSM states | Implicit | Implicit |

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with intention by [Hidden Leaf Networks](https://hiddenleafnetworks.com) — an applied AI studio.

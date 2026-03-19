# Getting Started

## Installation

```bash
pip install aria-core
```

### With LLM providers

```bash
pip install aria-core[openai]          # OpenAI / xAI (Grok)
pip install aria-core[anthropic]       # Anthropic Claude
pip install aria-core[all-providers]   # Everything
```

### From source

```bash
git clone https://github.com/Hidden-Leaf-Networks/aria-core.git
cd aria-core
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- pydantic >= 2.5.0

## Quick Start

The fastest way to see Aria Core in action — no API keys needed:

```python
import asyncio
from aria_core import AgentStateMachine, AgentConfig, Router
from aria_core.adapters import OpenAIAdapterStub

async def main():
    machine = AgentStateMachine(
        router=Router(),
        planner=StubPlanner(),  # see examples/basic_agent.py
        executor=StubExecutor(),
        adapter=OpenAIAdapterStub(),
        config=AgentConfig(max_steps=10),
    )

    result = await machine.process_message("What is machine learning?")
    print(f"State: {result.state}")      # "complete"
    print(f"Response: {result.response}") # "[STUB] Response to: ..."

asyncio.run(main())
```

Run the included examples for fully working demos:

```bash
python examples/basic_agent.py     # FSM pipeline with event tracing
python examples/risk_scoring.py    # Risk calculation + approval workflow
```

## Core Concepts

### 1. Deterministic State Machine

Every agent runs through a finite state machine with validated transitions:

```
IDLE → ROUTING → PLANNING → EXECUTING_STEP → RESPONDING → COMPLETE
                    ↓              ↓
              (skip if direct)  AWAITING_APPROVAL
                                   ↓
                              (approve/reject)
```

No uncontrolled loops. Every transition is validated, logged, and inspectable.

### 2. Protocol-Based Architecture

Aria Core uses Python `Protocol` classes for loose coupling. You implement what you need:

```python
class RouterProtocol(Protocol):
    async def route(self, context: AgentContext) -> dict[str, Any]: ...

class PlannerProtocol(Protocol):
    async def create_plan(self, context: AgentContext) -> Any: ...

class ExecutorProtocol(Protocol):
    async def execute_step(self, context: AgentContext) -> dict[str, Any]: ...

class AdapterProtocol(Protocol):
    async def generate_response(self, context: AgentContext) -> str: ...
```

### 3. Permission-First Safety

Every action is risk-scored before execution. High-risk actions require approval:

```python
from aria_core import RiskEngine, SkillCategory, ImpactScope, RiskScoreInput

engine = RiskEngine()
score = engine.calculate(RiskScoreInput(
    skill_name="deploy_production",
    skill_category=SkillCategory.EXEC,
    impact_scope=ImpactScope.SYSTEM,
))
# score.score = 38, score.level = "low", score.requires_approval = False
```

### 4. Multi-Model Consensus

For high-stakes decisions, Deep Bridge queries multiple LLMs and requires consensus:

```python
from aria_core import DeepBridgeValidator, ConsensusMode

bridge = DeepBridgeValidator(
    consensus_mode=ConsensusMode.MAJORITY,
    min_votes=2,
)
result = await bridge.validate(
    action_id=uuid4(), plan_id=uuid4(),
    skill_name="execute_trade",
    args={"symbol": "BTC", "amount": 1000},
    risk_score=75, risk_level="high",
)
# result.consensus_reached, result.consensus_approved
```

## Next Steps

- [Architecture Guide](architecture.md) — How the modules fit together
- [API Reference](api-reference.md) — Complete class and method reference
- [Examples](examples.md) — Walkthrough of included examples

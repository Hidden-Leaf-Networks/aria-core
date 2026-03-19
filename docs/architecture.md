# Architecture

## System Overview

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

## Module Map

| Module | Package | Purpose |
|--------|---------|---------|
| Runtime | `aria_core.runtime` | FSM state machine, models, transitions |
| Router | `aria_core.router` | Intent classification, strategy routing |
| Orchestration | `aria_core.orchestration` | Deep Bridge multi-model consensus |
| Planning | `aria_core.planning` | Plan lifecycle, dependency DAG, execution |
| Permissions | `aria_core.permissions` | Risk scoring, approval engine, audit |
| Adapters | `aria_core.adapters` | LLM provider integrations |

## Runtime: The State Machine

The `AgentStateMachine` is the execution backbone. It enforces a strict state graph — every transition must be in the valid set, or it's rejected.

### States

| State | Terminal | Purpose |
|-------|----------|---------|
| `IDLE` | No | Waiting for input |
| `ROUTING` | No | Classifying intent, selecting strategy |
| `PLANNING` | No | Generating execution plan |
| `EXECUTING_STEP` | No | Running a plan step |
| `AWAITING_APPROVAL` | No | Blocked on human approval |
| `RESPONDING` | No | Generating final LLM response |
| `COMPLETE` | Yes | Success terminal state |
| `ERROR` | Yes | Failure terminal state |

### Transition Graph

```
IDLE ──────────► ROUTING ──────────► PLANNING
                    │                    │
                    │                    ▼
                    │              EXECUTING_STEP ◄──┐
                    │                    │           │
                    │                    ├───────────┘ (next step)
                    │                    │
                    │                    ▼
                    │            AWAITING_APPROVAL
                    │                    │
                    │                    ▼
                    └──────────► RESPONDING
                                     │
                                     ▼
                                  COMPLETE

     (any non-terminal) ──────► ERROR
```

### Custom States

You can replace any state by passing a custom state registry:

```python
from aria_core.runtime.states import State, STATE_REGISTRY

class MyRoutingState(State):
    name = AgentStateEnum.ROUTING

    async def execute(self, machine, context):
        # Custom routing logic
        return AgentStateEnum.RESPONDING

custom_states = {**STATE_REGISTRY, AgentStateEnum.ROUTING: MyRoutingState}
machine = AgentStateMachine(..., states=custom_states)
```

### Event Observability

Every state transition and lifecycle event fires a callback:

```python
async def on_event(event_type: str, payload: dict):
    print(f"[{event_type}] {payload}")

machine = AgentStateMachine(..., event_callback=on_event)
```

Events emitted:
- `agent.start`, `agent.complete`, `agent.error`
- `routing.start`, `routing.complete`
- `planning.start`, `planning.complete`
- `step.start`, `step.complete`
- `approval.required`, `approval.granted`, `approval.denied`
- `responding.start`, `responding.complete`
- `transition.complete`, `transition.invalid`

## Router: Intent Classification

The `Router` scores incoming messages against a set of `RoutingStrategy` implementations and picks the highest-confidence match.

### Built-in Strategies

| Strategy | Triggers On | Intent |
|----------|------------|--------|
| `DirectStrategy` | "what is", "explain", simple questions | `simple_query` |
| `PlanStrategy` | "search for", "create", URLs, multi-step | `complex_task` |
| `ClarifyStrategy` | Empty, very short, or ambiguous input | `unclear` |

### Custom Strategies

```python
from aria_core.router.strategies import RoutingStrategy, RouteResult

class MyStrategy(RoutingStrategy):
    name = "my_strategy"

    async def should_use(self, context):
        # Return confidence 0-1
        return 0.9 if "deploy" in context.messages[-1].content else 0.0

    async def get_route(self, context):
        return RouteResult(
            strategy=self.name,
            intent="deployment",
            confidence=0.9,
            tools_needed=["deploy_service"],
        )

router = Router()
router.add_strategy(MyStrategy())
```

## Deep Bridge: Multi-Model Consensus

For high-stakes actions, Deep Bridge queries multiple LLM providers in parallel and computes consensus.

### Consensus Modes

| Mode | Requirement |
|------|-------------|
| `UNANIMOUS` | All models must approve |
| `MAJORITY` | >50% must approve |
| `ANY` | At least one approves |

### Provider Architecture

Built-in providers query OpenAI, Anthropic, and xAI APIs. Custom providers implement `ModelQueryProvider`:

```python
class ModelQueryProvider(Protocol):
    async def query(self, model: str, prompt: str) -> str: ...
```

### Auto-Detection

If no providers are configured, Deep Bridge auto-detects available API keys from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`).

## Planning: Dependency DAG Execution

The `PlanEngine` manages plan lifecycle from draft through execution.

### Plan States

```
DRAFT → PLANNED → QUEUED → EXECUTING → COMPLETED
                              │
                           BLOCKED → (resume) → EXECUTING
                              │
                           FAILED
```

### Dependency Validation

Actions can declare dependencies on other actions by index. The engine validates:
- No circular dependencies (DFS cycle detection)
- All referenced indices exist
- No self-references

```python
engine = PlanEngine(skill_executor=my_executor)

plan = engine.create_plan("research", actions=[
    {"name": "search", "skill_name": "web_search", "skill_args": {"q": "AI safety"}},
    {"name": "summarize", "skill_name": "summarize", "dependencies": [0]},  # waits for search
    {"name": "report", "skill_name": "generate_doc", "dependencies": [1]},  # waits for summary
])

plan = engine.validate_plan(plan.id)
plan = engine.start_plan(plan.id)
plan = await engine.execute_all(plan.id)
```

## Permissions: Risk Scoring & Approval Gates

### Risk Scoring Algorithm

Every action gets a deterministic score from 0-100:

| Factor | Weight | Range |
|--------|--------|-------|
| Skill category (READ/WRITE/EXEC/EXTERNAL) | 40% | 0-40 |
| Impact scope (LOCAL/USER/SYSTEM/EXTERNAL) | 30% | 0-30 |
| Historical penalties (failures + violations) | 10% | 0-10 |
| Context modifiers (sensitive args, network, etc.) | 20% | 0-20 |

Risk levels: **safe** (0-20), **low** (20-40), **medium** (40-60), **high** (60-80), **critical** (80-100)

### Approval Gates

When a score exceeds the policy threshold, the `ApprovalEngine` creates an approval request:

```python
from aria_core.permissions import ApprovalEngine, ApprovalGate

engine = ApprovalEngine(gates=[
    ApprovalGate(name="high_risk", risk_threshold=60, required_approvers=1),
    ApprovalGate(name="critical", risk_threshold=80, required_approvers=2),
])

if engine.requires_approval(risk_score=75):
    approval = engine.create_approval(plan_id=plan.id, risk_score=score)
    # Wait for human decision
    response = engine.approve(approval.id, approver_id="admin-1", reason="Reviewed")
```

### Custom Policies

```python
from aria_core.permissions.models import RiskPolicy

strict = RiskPolicy(name="strict", approval_threshold=20)
engine = RiskEngine(policy=strict)
```

## Adapters: LLM Providers

All adapters implement `ModelAdapter` and provide:
- `generate_response()` — Single response
- `stream_response()` — Token-by-token streaming
- `generate_with_tools()` — Function/tool calling

| Adapter | Provider | Default Model | Install Extra |
|---------|----------|---------------|---------------|
| `OpenAIAdapter` | OpenAI | gpt-4 | `[openai]` |
| `AnthropicAdapter` | Anthropic | claude-sonnet-4-20250514 | `[anthropic]` |
| `XAIAdapter` | xAI (Grok) | grok-2-latest | `[openai]` |
| `OpenAIAdapterStub` | None (testing) | — | — |

### Anthropic Extended Thinking

```python
config = AgentConfig(
    extended_thinking=True,
    thinking_budget_tokens=5000,
)
adapter = AnthropicAdapter(api_key="...")
response = await adapter.generate_response(AgentContext(config=config, messages=[...]))
```

### Custom Adapters

```python
from aria_core.adapters import ModelAdapter

class MyAdapter(ModelAdapter):
    name = "my_provider"

    async def generate_response(self, context):
        messages = self.format_messages(context)
        # Call your LLM API
        return "response"
```

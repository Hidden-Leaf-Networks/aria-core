# API Reference

## Package Exports

Everything below is importable from the top-level package:

```python
from aria_core import (
    # Runtime
    AgentStateMachine, AgentConfig, AgentContext, AgentResult,
    AgentStateEnum, ChatMessage, MessageRole,
    State, STATE_REGISTRY, Transition,
    # Orchestration
    DeepBridgeValidator, ConsensusMode,
    # Router
    Router, RoutingStrategy, RouteResult,
    # Permissions
    RiskEngine, SkillCategory, ImpactScope, RiskScoreInput,
    # Planning
    PlanEngine,
    # Adapters
    ModelAdapter, OpenAIAdapter, OpenAIAdapterStub,
    AnthropicAdapter, XAIAdapter,
)
```

---

## Runtime

### `AgentStateMachine`

The core execution engine. Runs a finite state machine from IDLE to COMPLETE/ERROR.

```python
AgentStateMachine(
    router: RouterProtocol,
    planner: PlannerProtocol,
    executor: ExecutorProtocol,
    adapter: AdapterProtocol,
    config: AgentConfig | None = None,
    event_callback: EventCallback | None = None,
    states: dict[AgentStateEnum, type[State]] | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `await process_message(message, conversation_id?, config?, metadata?)` | `AgentResult` | Process a user message end-to-end |
| `await run(context)` | `AgentResult` | Run FSM on an existing context |
| `await transition_to(next_state, context)` | `bool` | Attempt a state transition |
| `reset()` | `None` | Reset to IDLE |
| `await emit_event(event_type, payload)` | `None` | Fire event callback |

| Property | Type | Description |
|----------|------|-------------|
| `current_state` | `AgentStateEnum` | Current FSM state |
| `is_terminal` | `bool` | True if COMPLETE or ERROR |
| `transition_history` | `list[TransitionResult]` | All transitions taken |

### `AgentConfig`

```python
AgentConfig(
    max_steps: int = 10,              # 1-50
    max_tokens: int = 4096,
    timeout_seconds: int = 300,       # 1-600
    model: str = "gpt-4",
    temperature: float | None = 0.7,  # 0.0-2.0
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    require_approval_for_restricted: bool = True,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 10000,  # 1024-128000
    use_llm_planning: bool = False,
    self_critique: bool = False,
)
```

### `AgentContext`

```python
AgentContext(
    id: UUID,                          # auto-generated
    conversation_id: UUID,             # auto-generated
    config: AgentConfig,               # defaults to AgentConfig()
    messages: list[ChatMessage],
    current_plan_id: UUID | None,
    current_step_index: int = 0,
    step_count: int = 0,
    skill_results: dict[str, dict],
    metadata: dict[str, Any],
    created_at: datetime,
)
```

### `AgentResult`

```python
AgentResult(
    id: UUID,
    state: AgentStateEnum,
    context: AgentContext,
    error: str | None,
    created_at: datetime,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `is_terminal` | `bool` | True if COMPLETE or ERROR |
| `response` | `str | None` | Agent's response text from metadata |

### `AgentStateEnum`

`IDLE`, `ROUTING`, `PLANNING`, `EXECUTING_STEP`, `AWAITING_APPROVAL`, `RESPONDING`, `COMPLETE`, `ERROR`

### `ChatMessage`

```python
ChatMessage(role: MessageRole, content: str, name?: str, tool_call_id?: str)
```

### `MessageRole`

`USER`, `ASSISTANT`, `SYSTEM`, `TOOL`

### `State` (Abstract)

Base class for FSM state implementations.

```python
class State(ABC):
    name: AgentStateEnum
    is_terminal: bool = False

    async def enter(self, machine, context) -> None
    async def execute(self, machine, context) -> AgentStateEnum
    async def exit(self, machine, context) -> None
```

### `Transition`

Static methods for transition validation.

| Method | Returns | Description |
|--------|---------|-------------|
| `Transition.is_valid(from_state, to_state)` | `bool` | Check if transition is allowed |
| `Transition.validate(from_state, to_state)` | `TransitionResult` | Validate with error details |
| `Transition.get_allowed_transitions(state)` | `set[AgentStateEnum]` | Valid next states |
| `Transition.is_terminal(state)` | `bool` | Check if state is terminal |

---

## Router

### `Router`

```python
Router(strategies: list[RoutingStrategy] | None = None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `await route(context)` | `dict` | Route to best strategy. Returns `{strategy, intent, confidence, tools_needed, metadata, all_scores}` |
| `add_strategy(strategy)` | `None` | Register a strategy |
| `remove_strategy(name)` | `bool` | Remove by name |

### `RoutingStrategy` (Abstract)

```python
class RoutingStrategy(ABC):
    name: str

    async def should_use(self, context: AgentContext) -> float  # 0-1
    async def get_route(self, context: AgentContext) -> RouteResult
```

### `RouteResult`

```python
RouteResult(
    strategy: str,
    intent: str,
    confidence: float,
    tools_needed: list[str] = [],
    metadata: dict = {},
)
```

### Built-in Strategies

- **`DirectStrategy`** — Simple questions → `simple_query`
- **`PlanStrategy`** — Multi-step tasks, URLs, tool keywords → `complex_task` / `plan_creation`
- **`ClarifyStrategy`** — Ambiguous or empty input → `unclear`

---

## Orchestration

### `DeepBridgeValidator`

```python
DeepBridgeValidator(
    consensus_mode: ConsensusMode = ConsensusMode.MAJORITY,
    min_votes: int = 2,
    timeout_seconds: int = 30,
    models: list[str] | None = None,
    providers: dict[str, ModelQueryProvider] | None = None,
    prompt_builder: Callable[..., str] | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `await validate(action_id, plan_id, skill_name, args, risk_score, risk_level, context?)` | `DeepBridgeResult` | Query models and compute consensus |

### `ConsensusMode`

`UNANIMOUS`, `MAJORITY`, `ANY`

### `DeepBridgeResult`

```python
DeepBridgeResult(
    id: UUID,
    action_id: UUID,
    plan_id: UUID,
    consensus_reached: bool,
    consensus_approved: bool,
    consensus_mode: ConsensusMode,
    votes: list[ModelVote],
    total_votes: int,
    approve_votes: int,
    reject_votes: int,
    average_confidence: float,
    min_confidence: float,
    total_time_ms: int,
    skill_name: str,
    risk_score: int,
)
```

### `ModelVote`

```python
ModelVote(
    model_name: str,
    provider: str,
    approved: bool,
    confidence: float,
    reasoning: str,
    response_time_ms: int,
    timestamp: datetime,
)
```

### `ModelQueryProvider` (Protocol)

```python
class ModelQueryProvider(Protocol):
    async def query(self, model: str, prompt: str) -> str: ...
```

---

## Permissions

### `RiskEngine`

```python
RiskEngine(policy: RiskPolicy | None = None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `calculate(input)` | `RiskScore` | Full deterministic risk scoring |
| `evaluate_quick(skill_category, impact_scope)` | `tuple[int, str, bool]` | Quick score without full input |
| `calculate_aggregate_risk(risk_scores)` | `tuple[int, str, bool]` | Weighted average for plans |
| `update_policy(policy)` | `None` | Swap risk policy |

### `RiskScoreInput`

```python
RiskScoreInput(
    skill_name: str,
    skill_category: SkillCategory,
    impact_scope: ImpactScope,
    historical_failures: int = 0,
    historical_violations: int = 0,
    is_first_execution: bool = False,
    has_sensitive_args: bool = False,
    targets_external_system: bool = False,
    requires_network: bool = False,
    modifies_persistent_state: bool = False,
)
```

### `RiskScore`

```python
RiskScore(
    id: UUID,
    score: int,                  # 0-100
    level: str,                  # safe/low/medium/high/critical
    requires_approval: bool,
    factors: list[RiskFactor],
    input_hash: str,             # SHA256 for reproducibility
    calculated_at: datetime,
    calculation_version: str,
)
```

### `SkillCategory`

| Value | Risk Weight |
|-------|-------------|
| `READ` | 0.1 |
| `WRITE` | 0.3 |
| `EXTERNAL` | 0.4 |
| `EXEC` | 0.5 |

### `ImpactScope`

| Value | Risk Weight |
|-------|-------------|
| `LOCAL` | 0.1 |
| `USER` | 0.3 |
| `EXTERNAL` | 0.5 |
| `SYSTEM` | 0.6 |

### `RiskPolicy`

```python
RiskPolicy(
    name: str,
    approval_threshold: int = 50,    # 0-100
    block_threshold: int = 80,       # 0-100
    first_execution_modifier: float = 1.2,
    failure_history_modifier: float = 0.05,
    violation_history_modifier: float = 0.1,
)
```

### `ApprovalEngine`

```python
ApprovalEngine(
    gates: list[ApprovalGate] | None = None,
    default_timeout_minutes: int = 60,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate_gates(risk_score)` | `ApprovalGate | None` | First triggered gate |
| `requires_approval(risk_score)` | `bool` | Quick check |
| `create_approval(plan_id, risk_score, action_id?, context?)` | `Approval` | Create approval request |
| `get_approval(approval_id)` | `Approval | None` | Retrieve by ID |
| `get_approval_for_plan(plan_id)` | `Approval | None` | Get pending approval |
| `decide(approval_id, decision, approver_id, approver_type?, reason?)` | `ApprovalResponse` | Record decision |
| `approve(approval_id, approver_id, reason?)` | `ApprovalResponse` | Approve shorthand |
| `reject(approval_id, approver_id, reason?)` | `ApprovalResponse` | Reject shorthand |
| `verify_approved(plan_id, action_id?)` | `bool` | Verify (raises if denied/expired) |
| `check_and_expire()` | `list[Approval]` | Expire stale approvals |
| `get_pending_approvals()` | `list[Approval]` | All pending |
| `get_pending_summary()` | `PendingApprovalSummary` | Summary stats |
| `add_gate(gate)` | `None` | Register gate |
| `remove_gate(gate_id)` | `bool` | Remove gate |

### Exceptions

| Exception | Attributes |
|-----------|-----------|
| `ApprovalRequiredError` | `approval_id`, `gate_name`, `risk_score` |
| `ApprovalDeniedError` | `approval_id`, `reason` |
| `ApprovalExpiredError` | `approval_id`, `expired_at` |
| `ApprovalNotFoundError` | `approval_id` |
| `InvalidApprovalStateError` | `approval_id` |
| `UnauthorizedApproverError` | `approver_id`, `gate_name` |

---

## Planning

### `PlanEngine`

```python
PlanEngine(
    skill_executor: SkillExecutor | None = None,
    event_callback: Callable[[str, dict], Any] | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `create_plan(name, actions, description?, conversation_id?, prompt?, created_by?)` | `Plan` | Create in DRAFT state |
| `get_plan(plan_id)` | `Plan | None` | Retrieve plan |
| `validate_plan(plan_id)` | `Plan` | Validate deps, DRAFT → PLANNED |
| `start_plan(plan_id)` | `Plan` | PLANNED → QUEUED → EXECUTING |
| `await execute_next_action(plan_id)` | `Plan` | Execute next ready action |
| `await execute_all(plan_id)` | `Plan` | Run all actions to completion |

### `Plan`

```python
Plan(
    id: UUID,
    name: str,
    description: str = "",
    state: PlanState = DRAFT,
    actions: list[PlanAction],
    version: int = 1,
    ...
)
```

| Property | Type | Description |
|----------|------|-------------|
| `is_terminal` | `bool` | COMPLETED, FAILED, or ARCHIVED |
| `progress` | `float` | 0.0–1.0 completion ratio |

### `PlanAction`

```python
PlanAction(
    name: str,
    skill_name: str | None = None,
    skill_args: dict | None = None,
    dependencies: list[int] = [],   # indices of prerequisite actions
)
```

### `PlanState`

`DRAFT`, `PLANNED`, `QUEUED`, `EXECUTING`, `BLOCKED`, `COMPLETED`, `FAILED`, `ARCHIVED`

### `ActionState`

`PENDING`, `QUEUED`, `EXECUTING`, `AWAITING_APPROVAL`, `APPROVED`, `COMPLETED`, `FAILED`, `SKIPPED`

### `ExecutionResult`

```python
ExecutionResult(
    success: bool,
    result: dict | None = None,
    error: str | None = None,
    execution_time_ms: int | None = None,
)
```

### `SkillExecutor` (Type Alias)

```python
SkillExecutor = Callable[[str, dict | None], Awaitable[ExecutionResult]]
```

---

## Adapters

### `ModelAdapter` (Abstract)

```python
class ModelAdapter(ABC):
    name: str
    supports_streaming: bool = True
    supports_tools: bool = True
```

| Method | Returns | Description |
|--------|---------|-------------|
| `await generate_response(context)` | `str` | Generate single response |
| `await stream_response(context)` | `AsyncIterator[str]` | Stream tokens |
| `await generate_with_tools(context, tools)` | `tuple[str, list | None]` | Response + tool calls |
| `format_messages(context)` | `list[dict]` | Convert context to API format |

### `OpenAIAdapter`

```python
OpenAIAdapter(api_key?: str, model: str = "gpt-4", base_url?: str)
```

### `AnthropicAdapter`

```python
AnthropicAdapter(api_key?: str, model: str = "claude-sonnet-4-20250514")
```

`format_messages()` returns `tuple[str, list[dict]]` (system prompt separated).

Supports `extended_thinking` via `AgentConfig`.

### `XAIAdapter`

```python
XAIAdapter(api_key?: str, model: str = "grok-2-latest")
```

Extends `OpenAIAdapter` with xAI's API endpoint (`https://api.x.ai/v1`).

### `OpenAIAdapterStub`

No-op adapter for testing. Returns `[STUB] Response to: {message}`.

# Examples

Aria Core ships with two runnable examples that require no API keys.

## Basic Agent (`examples/basic_agent.py`)

Demonstrates the full FSM pipeline with event tracing.

```bash
python examples/basic_agent.py
```

### What it shows

1. **Direct response flow** — A simple question ("What is machine learning?") routes through `DirectStrategy`, skips planning, and goes straight to the LLM response.

2. **Plan-triggered flow** — A complex query ("Search for AI papers and create a summary") triggers `PlanStrategy`, enters the planning state, then proceeds to response.

3. **Event observability** — Every state transition and lifecycle event is printed, showing the full audit trail.

### Key patterns

**Stub implementations** — The example uses `OpenAIAdapterStub` for the LLM and minimal stub classes for the planner and executor. This is the recommended approach for testing:

```python
from aria_core.adapters import OpenAIAdapterStub

class StubPlanner:
    async def create_plan(self, context):
        # Return a plan with no steps → triggers direct response
        return Plan(steps=[])

class StubExecutor:
    async def execute_step(self, context):
        return {"success": True}
```

**Event callbacks** — Attach a callback to observe everything the agent does:

```python
async def on_event(event_type: str, payload: dict):
    print(f"[{event_type}] {payload}")

machine = AgentStateMachine(..., event_callback=on_event)
```

## Risk Scoring (`examples/risk_scoring.py`)

Demonstrates deterministic risk calculation and the approval workflow.

```bash
python examples/risk_scoring.py
```

### What it shows

1. **Default policy scoring** — Scores four actions (read_file, write_database, send_email, execute_code) under the default policy. Shows how skill category and impact scope affect the score.

2. **Custom strict policy** — Creates a policy with `approval_threshold=20` and re-scores the same actions. Actions that were safe under the default policy now require approval.

3. **Approval workflow** — Creates a high-risk action, generates an approval request, and approves it programmatically.

### Key patterns

**Quick scoring:**

```python
engine = RiskEngine()
score = engine.calculate(RiskScoreInput(
    skill_name="read_file",
    skill_category=SkillCategory.READ,
    impact_scope=ImpactScope.LOCAL,
))
# score.score = 7, score.level = "safe"
```

**Custom policies:**

```python
strict = RiskPolicy(name="strict", approval_threshold=20)
engine = RiskEngine(policy=strict)
```

**Approval flow:**

```python
approval_engine = ApprovalEngine()
approval = approval_engine.create_approval(plan_id=uuid4(), risk_score=score)
response = approval_engine.approve(approval.id, approver_id="admin-1", reason="Reviewed")
```

## Running with Real LLMs

To use a real LLM provider, swap the stub adapter:

```python
from aria_core.adapters import OpenAIAdapter

machine = AgentStateMachine(
    router=Router(),
    planner=your_planner,
    executor=your_executor,
    adapter=OpenAIAdapter(api_key="sk-..."),
    config=AgentConfig(model="gpt-4"),
)

result = await machine.process_message("Explain quantum computing")
print(result.response)  # Real LLM response
```

For Anthropic:

```python
from aria_core.adapters import AnthropicAdapter

adapter = AnthropicAdapter(api_key="sk-ant-...")
# Extended thinking:
config = AgentConfig(extended_thinking=True, thinking_budget_tokens=5000)
```

For xAI (Grok):

```python
from aria_core.adapters import XAIAdapter

adapter = XAIAdapter(api_key="xai-...")
```

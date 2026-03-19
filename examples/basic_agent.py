"""Basic agent example — runs the full FSM pipeline with a stub adapter.

No API keys required. Demonstrates:
- Agent state machine with all transitions
- Intent routing (direct vs plan strategies)
- Event observability

Run:
    python examples/basic_agent.py
"""

import asyncio

from aria_core import AgentStateMachine, AgentConfig, Router
from aria_core.adapters import OpenAIAdapterStub


# Simple router that satisfies the RouterProtocol
router = Router()


# Planner stub — returns a plan with no steps (triggers direct response)
class StubPlanner:
    async def create_plan(self, context):
        from pydantic import BaseModel, Field
        from uuid import uuid4

        class Plan(BaseModel):
            id: object = Field(default_factory=uuid4)
            steps: list = Field(default_factory=list)
            def model_dump(self, **kw): return {"id": str(self.id), "steps": []}

        return Plan()


# Executor stub
class StubExecutor:
    async def execute_step(self, context):
        return {"success": True}


async def main():
    events = []

    async def on_event(event_type: str, payload: dict):
        events.append(event_type)
        print(f"  [{event_type}] {payload}")

    machine = AgentStateMachine(
        router=router,
        planner=StubPlanner(),
        executor=StubExecutor(),
        adapter=OpenAIAdapterStub(),
        config=AgentConfig(max_steps=10),
        event_callback=on_event,
    )

    print("=== Aria Core — Basic Agent ===\n")

    # Process a simple query
    print("Query: 'What is machine learning?'")
    result = await machine.process_message("What is machine learning?")

    print(f"\nFinal state: {result.state}")
    print(f"Response: {result.response}")
    print(f"Events fired: {len(events)}")
    print(f"Transitions: {len(machine.transition_history)}")

    # Process a complex query
    print("\n---\n")
    machine.reset()
    events.clear()

    print("Query: 'Search for AI papers and create a summary'")
    result = await machine.process_message("Search for AI papers and create a summary")

    print(f"\nFinal state: {result.state}")
    print(f"Response: {result.response}")
    print(f"Events fired: {len(events)}")


if __name__ == "__main__":
    asyncio.run(main())

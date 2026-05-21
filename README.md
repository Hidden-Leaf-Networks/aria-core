# Aria Core

**Multi-tenant AI agent framework with deterministic execution, white-label SaaS, and permission-first safety.**

Built by [Hidden Leaf Networks](https://hiddenleafnetworks.com).

> Aria Core is the open-source foundation extracted from [Aria](https://hiddenleafnetworks.com/portfolio) — a production agent system running 14 autonomous agents with governed skills, multi-model consensus, and full audit trails.

---

## What is Aria Core?

Aria Core is a Python framework for building and deploying AI agent platforms that are **deterministic, safe, multi-tenant, and production-ready**. Unlike prompt-chain frameworks, Aria Core provides:

- **Deterministic FSM** — 8-state machine with validated transitions, max-step enforcement, no uncontrolled loops
- **Multi-model consensus** — Deep Bridge queries multiple LLMs in parallel for high-stakes decisions
- **Permission-first safety** — risk scoring (0-100), approval gates with RBAC, immutable decision records
- **Multi-tenant isolation** — tenant-scoped persistence, config overrides, rate limiting, billing
- **Event sourcing** — append-only audit trail with full replay capability
- **White-label ready** — REST API, JWT auth, Config Portal, Helm chart, Stripe billing

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Config Portal (React 19 HUD)          │
├──────────────────────────────────────────────────────────┤
│  REST API (FastAPI)    JWT Auth │ RBAC │ WebSocket        │
├──────────────────────────────────────────────────────────┤
│  Tenant Layer          Isolation │ Config │ Guard          │
├──────────────────────────────────────────────────────────┤
│  Router                Intent classification & routing     │
├──────────────────────────────────────────────────────────┤
│  Planner               Plans, actions, dependencies       │
├──────────────────────────────────────────────────────────┤
│  Runtime (FSM)         IDLE → ROUTING → PLANNING →        │
│                        EXECUTING → RESPONDING → COMPLETE   │
├──────────────────────────────────────────────────────────┤
│  Permissions           Risk scoring │ Approval gates       │
├──────────────────────────────────────────────────────────┤
│  Deep Bridge           Multi-model consensus voting        │
├──────────────────────────────────────────────────────────┤
│  Persistence           InMemory │ PostgreSQL │ EventStore  │
├──────────────────────────────────────────────────────────┤
│  Billing               UsageMeter │ Stripe │ Pricing       │
├──────────────────────────────────────────────────────────┤
│  Adapters              OpenAI │ Anthropic │ xAI            │
└──────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core framework
pip install aria-core

# With LLM providers
pip install aria-core[openai]          # OpenAI only
pip install aria-core[anthropic]       # Anthropic only
pip install aria-core[all-providers]   # All providers

# With API server
pip install aria-core[api]             # FastAPI + JWT + WebSocket

# With PostgreSQL persistence
pip install aria-core[postgres]        # SQLAlchemy + asyncpg + Alembic

# Everything
pip install aria-core[all]
```

## Quick Start

### As a Library

```python
from aria_core import AgentStateMachine, AgentConfig, Router

# Create an agent with deterministic execution
machine = AgentStateMachine(
    router=my_router,
    planner=my_planner,
    executor=my_executor,
    adapter=my_llm_adapter,
    config=AgentConfig(max_steps=10, model="gpt-4"),
)

result = await machine.process_message("Analyze this dataset")
```

### As a Multi-Tenant API

```bash
# Start the API server
ARIA_JWT_SECRET=your-secret \
ARIA_PERSISTENCE=memory \
uvicorn aria_core.api:create_app --factory

# Create a tenant
curl -X POST http://localhost:8000/api/v1/tenants \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"slug": "acme-co", "name": "Acme Corporation"}'
```

### With Docker

```bash
cp .env.example .env
# Edit .env with your JWT secret and database credentials
docker-compose up
```

## API Endpoints (27 routes)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | - | Liveness probe |
| `GET` | `/ready` | - | Readiness probe |
| `GET` | `/.well-known/jwks.json` | - | RS256 public keys |
| `GET` | `/api/v1/pricing/tiers` | - | Pricing tiers |
| `GET` | `/api/v1/pricing/calculate` | - | Cost calculator |
| `POST` | `/api/v1/tenants` | Admin | Create tenant |
| `GET` | `/api/v1/tenants` | Admin | List tenants |
| `GET` | `/api/v1/tenants/{id}` | Auth | Get tenant |
| `PUT` | `/api/v1/tenants/{id}/config` | Admin | Update config |
| `POST` | `/api/v1/plans` | Operator | Create plan |
| `GET` | `/api/v1/plans` | Auth | List plans |
| `GET` | `/api/v1/plans/{id}` | Auth | Get plan |
| `DELETE` | `/api/v1/plans/{id}` | Operator | Delete plan |
| `GET` | `/api/v1/approvals` | Auth | List approvals |
| `GET` | `/api/v1/approvals/{id}` | Auth | Get approval |
| `POST` | `/api/v1/approvals/{id}/approve` | Operator | Approve |
| `POST` | `/api/v1/approvals/{id}/reject` | Operator | Reject |
| `GET` | `/api/v1/agents` | Auth | List agents |
| `POST` | `/api/v1/agents` | Operator | Register agent |
| `GET` | `/api/v1/agents/{id}` | Auth | Get agent |
| `DELETE` | `/api/v1/agents/{id}` | Operator | Delete agent |
| `GET` | `/api/v1/archetypes` | Auth | List archetypes |
| `POST` | `/api/v1/archetypes` | Operator | Create archetype |
| `POST` | `/api/v1/archetypes/{id}/deploy` | Operator | Deploy archetype |
| `GET` | `/api/v1/events` | Auth | List events |
| `GET` | `/api/v1/events/replay` | Auth | Replay events |
| `WS` | `/ws/events` | JWT | Real-time stream |

## Modules

| Module | Description |
|--------|-------------|
| `aria_core.runtime` | FSM state machine with deterministic execution |
| `aria_core.router` | Intent classification and strategy routing |
| `aria_core.orchestration` | Deep Bridge multi-model consensus |
| `aria_core.planning` | Plan lifecycle, dependencies, versioning |
| `aria_core.permissions` | Risk scoring, approval gates, audit |
| `aria_core.adapters` | LLM adapters (OpenAI, Anthropic, xAI) |
| `aria_core.tenant` | Multi-tenant isolation, config resolution |
| `aria_core.persistence` | InMemory + PostgreSQL providers, event store |
| `aria_core.api` | FastAPI REST API, JWT auth, WebSocket |
| `aria_core.billing` | Usage metering, Stripe adapter, pricing |
| `aria_core.archetypes` | Agent template registry |

## Why Aria Core?

| Feature | Aria Core | LangGraph | CrewAI | Letta |
|---------|-----------|-----------|--------|-------|
| Deterministic FSM | 8-state validated | No | No | No |
| Multi-model consensus | Deep Bridge voting | Basic routing | None | None |
| Risk scoring + approvals | 0-100 + gates | None | None | None |
| Multi-tenant | Full isolation | No | No | No |
| White-label | Config Portal + branding | No | No | No |
| Event sourcing | Replay + projections | Checkpoints | None | None |
| Usage billing | Stripe metered | Pricing page | None | None |
| Deployment | Docker + Helm + HPA | Cloud only | Docker | Docker |

## Production Heritage

Extracted from a production system running:

- **14 autonomous agents** across DevRel, research, outreach, acquisition, and operations
- **18 governed skills** across SAFE/LOW/MEDIUM/HIGH risk tiers
- **Multi-model orchestration** across OpenAI, Anthropic, and xAI
- **Full event store** with PostgreSQL persistence
- **Real-time WebSocket streaming** for observability

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

Built with intention by [Hidden Leaf Networks](https://hiddenleafnetworks.com) — an applied AI studio.

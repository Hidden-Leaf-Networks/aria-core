# Aria Core API Reference

> Comprehensive REST API documentation for Aria Core multi-tenant agent framework.
>
> **Base URL:** `https://your-instance.com`
> **OpenAPI Spec:** Available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when running the FastAPI server.
> **Version:** v1

---

## Table of Contents

1. [Authentication](#authentication)
2. [Tenants](#tenants)
3. [Plans](#plans)
4. [Approvals](#approvals)
5. [Agents](#agents)
6. [Archetypes](#archetypes)
7. [Providers](#providers)
8. [Events](#events)
9. [Contexts](#contexts)
10. [Execution](#execution)
11. [WebSocket](#websocket)
12. [Billing](#billing)
13. [Pricing](#pricing)
14. [Health](#health)
15. [Scheduling](#scheduling)
16. [Error Codes](#error-codes)

---

## Authentication

Aria Core uses JWT (JSON Web Token) authentication with role-based access control (RBAC). Tokens are validated using HS256 (shared secret) or RS256 (public key via JWKS).

### Token Claims

```json
{
  "sub": "user-123",
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_slug": "acme-co",
  "role": "operator",
  "iss": "aria-core",
  "aud": "aria-core-api",
  "exp": 1234567890
}
```

### Roles

| Role | Level | Permissions |
|------|-------|-------------|
| `viewer` | 0 | Read-only access to own tenant data |
| `operator` | 1 | Create/modify plans, agents, archetypes, approve/reject |
| `admin` | 2 | Full access: tenant CRUD, provider config, billing, all tenants |

Roles are hierarchical. An `admin` has all `operator` and `viewer` permissions.

### Header Format

All authenticated requests require the `Authorization` header:

```
Authorization: Bearer <jwt-token>
```

### Obtaining a Token

For development, create tokens using the `aria_core.api.auth.create_token` function:

```python
from aria_core.api.auth import create_token, Role
from uuid import UUID

token = create_token(
    user_id="user-123",
    tenant_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
    role=Role.OPERATOR,
    secret="your-jwt-secret",
    tenant_slug="acme-co",
    expires_in_seconds=3600,
)
```

For production, integrate with your identity provider (Auth0, Keycloak, etc.) and configure the `ARIA_JWT_SECRET` or RS256 JWKS endpoint.

### JWKS Endpoint (RS256)

```bash
curl https://your-instance.com/.well-known/jwks.json
```

```json
{
  "keys": [
    {
      "kty": "RSA",
      "kid": "aria-core-key-1",
      "use": "sig",
      "alg": "RS256",
      "n": "...",
      "e": "AQAB"
    }
  ]
}
```

---

## Tenants

Multi-tenant isolation is enforced at every layer. Each tenant has its own agents, plans, events, config, and billing.

### Create Tenant

**`POST /api/v1/tenants`** | Role: `admin`

```bash
curl -X POST https://your-instance.com/api/v1/tenants \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "acme-co",
    "name": "Acme Corporation",
    "config": {
      "default_model": "gpt-4o",
      "max_concurrent_agents": 20,
      "features": {
        "deep_bridge": true,
        "event_sourcing": true
      }
    }
  }'
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "slug": "acme-co",
  "name": "Acme Corporation",
  "config": {
    "display_name": null,
    "logo_url": null,
    "theme": {},
    "default_model": "gpt-4o",
    "allowed_models": [],
    "max_tokens": null,
    "max_concurrent_agents": 20,
    "max_plans_per_hour": 100,
    "max_events_per_day": 100000,
    "features": {
      "deep_bridge": true,
      "event_sourcing": true
    },
    "risk_policy_id": null,
    "approval_gates": [],
    "metadata": {}
  },
  "is_active": true,
  "created_at": "2026-05-20T12:00:00Z",
  "updated_at": "2026-05-20T12:00:00Z"
}
```

### List Tenants

**`GET /api/v1/tenants`** | Role: `admin`

```bash
curl https://your-instance.com/api/v1/tenants \
  -H "Authorization: Bearer $TOKEN"
```

**Response:** Array of tenant objects.

### Get Tenant

**`GET /api/v1/tenants/{tenant_id}`** | Role: `auth` (admin sees any, others see own)

```bash
curl https://your-instance.com/api/v1/tenants/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

### Update Tenant Config

**`PUT /api/v1/tenants/{tenant_id}/config`** | Role: `admin`

```bash
curl -X PUT https://your-instance.com/api/v1/tenants/550e8400-e29b-41d4-a716-446655440000/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "Acme AI Platform",
    "logo_url": "https://acme.com/logo.png",
    "default_model": "claude-sonnet-4-20250514",
    "allowed_models": ["gpt-4o", "claude-sonnet-4-20250514", "grok-3"],
    "max_concurrent_agents": 50,
    "max_plans_per_hour": 500,
    "features": {
      "deep_bridge": true,
      "event_sourcing": true,
      "archetypes": true
    }
  }'
```

Only fields included in the request body are updated (partial update via `exclude_unset`).

---

## Plans

Plans represent execution workflows with ordered actions, dependency tracking, and lifecycle state management.

### Plan States

```
DRAFT -> PLANNED -> QUEUED -> EXECUTING -> COMPLETED
                                 |
                                 v
                              BLOCKED -> EXECUTING (retry)
                                 |
                                 v
                              FAILED -> DRAFT (rework)
                                 |
                                 v
                              ARCHIVED
```

### Create Plan

**`POST /api/v1/plans`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/plans \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Pipeline",
    "description": "Extract, analyze, and summarize market data",
    "actions": [
      {
        "name": "Extract Data",
        "description": "Scrape competitor pricing pages",
        "skill_name": "web_scraper",
        "skill_args": {"urls": ["https://competitor.com/pricing"]},
        "dependencies": []
      },
      {
        "name": "Analyze Trends",
        "description": "Identify pricing patterns",
        "skill_name": "data_analysis",
        "skill_args": {"mode": "trend"},
        "dependencies": [0]
      },
      {
        "name": "Generate Report",
        "description": "Write executive summary",
        "skill_name": "content_writer",
        "skill_args": {"format": "pdf"},
        "dependencies": [1]
      }
    ]
  }'
```

**Response:**

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Research Pipeline",
  "description": "Extract, analyze, and summarize market data",
  "state": "draft",
  "actions": [
    {
      "id": "...",
      "plan_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "index": 0,
      "name": "Extract Data",
      "description": "Scrape competitor pricing pages",
      "skill_name": "web_scraper",
      "skill_args": {"urls": ["https://competitor.com/pricing"]},
      "dependencies": [],
      "state": "pending",
      "risk_score": null,
      "requires_approval": false,
      "result": null,
      "error": null,
      "started_at": null,
      "completed_at": null
    }
  ],
  "current_action_index": 0,
  "aggregate_risk_score": null,
  "requires_approval": false,
  "version": 1,
  "created_at": "2026-05-20T12:00:00Z",
  "updated_at": "2026-05-20T12:00:00Z",
  "created_by": "user-123"
}
```

### List Plans

**`GET /api/v1/plans`** | Role: `auth`

```bash
# List all plans
curl https://your-instance.com/api/v1/plans \
  -H "Authorization: Bearer $TOKEN"

# Filter by state
curl "https://your-instance.com/api/v1/plans?state=executing&limit=20&offset=0" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `state` | string | null | Filter by plan state |
| `limit` | int | 50 | Max results (1-200) |
| `offset` | int | 0 | Pagination offset |

### Get Plan

**`GET /api/v1/plans/{plan_id}`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/plans/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Authorization: Bearer $TOKEN"
```

### Delete Plan

**`DELETE /api/v1/plans/{plan_id}`** | Role: `operator`

```bash
curl -X DELETE https://your-instance.com/api/v1/plans/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "deleted": true,
  "plan_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

## Approvals

Approval gates enforce human-in-the-loop oversight for high-risk actions. Actions with risk scores above a tenant's threshold are automatically held for approval.

### List Approvals

**`GET /api/v1/approvals`** | Role: `auth`

```bash
# All approvals
curl https://your-instance.com/api/v1/approvals \
  -H "Authorization: Bearer $TOKEN"

# Filter by state and plan
curl "https://your-instance.com/api/v1/approvals?state=pending&plan_id=a1b2c3d4-...&limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `state` | string | null | Filter: `pending`, `approved`, `rejected` |
| `plan_id` | UUID | null | Filter by plan |
| `limit` | int | 50 | Max results (1-200) |
| `offset` | int | 0 | Pagination offset |

### Get Approval

**`GET /api/v1/approvals/{approval_id}`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/approvals/b2c3d4e5-f6a7-8901-bcde-f12345678901 \
  -H "Authorization: Bearer $TOKEN"
```

### Approve

**`POST /api/v1/approvals/{approval_id}/approve`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/approvals/b2c3d4e5-f6a7-8901-bcde-f12345678901/approve \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "state": "approved",
  "decisions": [
    {
      "approval_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "decision": "approved",
      "approver_id": "user-123",
      "approver_type": "user",
      "reason": "Approved via Config Portal"
    }
  ],
  "resolved_at": "2026-05-20T12:05:00Z"
}
```

### Reject

**`POST /api/v1/approvals/{approval_id}/reject`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/approvals/b2c3d4e5-f6a7-8901-bcde-f12345678901/reject \
  -H "Authorization: Bearer $TOKEN"
```

---

## Agents

Agents are registered execution units within a tenant. Each agent has a model, system prompt, allowed skills, and execution limits.

### Register Agent

**`POST /api/v1/agents`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Analyst",
    "slug": "research-analyst",
    "description": "Autonomous research agent for market analysis",
    "model": "gpt-4o",
    "system_prompt": "You are a senior market research analyst...",
    "allowed_skills": ["web_search", "data_analysis", "content_writer"],
    "max_steps": 15,
    "temperature": 0.3
  }'
```

**Response:**

```json
{
  "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Research Analyst",
  "slug": "research-analyst",
  "description": "Autonomous research agent for market analysis",
  "model": "gpt-4o",
  "system_prompt": "You are a senior market research analyst...",
  "allowed_skills": ["web_search", "data_analysis", "content_writer"],
  "max_steps": 15,
  "temperature": 0.3,
  "status": "active",
  "executions": 0,
  "created_at": "2026-05-20T12:00:00Z",
  "created_by": "user-123"
}
```

### List Agents

**`GET /api/v1/agents`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/agents \
  -H "Authorization: Bearer $TOKEN"
```

### Get Agent

**`GET /api/v1/agents/{agent_id}`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/agents/c3d4e5f6-a7b8-9012-cdef-123456789012 \
  -H "Authorization: Bearer $TOKEN"
```

### Delete Agent

**`DELETE /api/v1/agents/{agent_id}`** | Role: `operator`

```bash
curl -X DELETE https://your-instance.com/api/v1/agents/c3d4e5f6-a7b8-9012-cdef-123456789012 \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "deleted": true,
  "agent_id": "c3d4e5f6-a7b8-9012-cdef-123456789012"
}
```

---

## Archetypes

Archetypes are reusable agent configuration templates. Deploy an archetype to instantly create a pre-configured agent.

### Categories

| Category | Description |
|----------|-------------|
| `research` | Research and analysis agents |
| `engineering` | Code generation, review, debugging |
| `content` | Writing, editing, social media |
| `data` | Data processing, ETL, visualization |
| `support` | Customer support, helpdesk |
| `security` | Security analysis, vulnerability scanning |
| `operations` | DevOps, monitoring, automation |
| `custom` | User-defined archetypes |

### List Archetypes

**`GET /api/v1/archetypes`** | Role: `auth`

```bash
# All archetypes
curl https://your-instance.com/api/v1/archetypes \
  -H "Authorization: Bearer $TOKEN"

# Filter by category
curl "https://your-instance.com/api/v1/archetypes?category=research" \
  -H "Authorization: Bearer $TOKEN"
```

### Get Archetype

**`GET /api/v1/archetypes/{archetype_id}`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/archetypes/d4e5f6a7-b8c9-0123-defa-234567890123 \
  -H "Authorization: Bearer $TOKEN"
```

### Create Archetype

**`POST /api/v1/archetypes`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/archetypes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DevRel Writer",
    "slug": "devrel-writer",
    "description": "Technical content creation for developer audiences",
    "category": "content",
    "icon": "pencil",
    "model": "claude-sonnet-4-20250514",
    "system_prompt": "You are a developer relations content writer...",
    "temperature": 0.7,
    "max_steps": 12,
    "allowed_skills": ["content_writer", "code_formatter", "seo_optimizer"],
    "require_approval": false,
    "tags": ["devrel", "content", "writing"]
  }'
```

**Response:**

```json
{
  "id": "d4e5f6a7-b8c9-0123-defa-234567890123",
  "tenant_id": null,
  "name": "DevRel Writer",
  "slug": "devrel-writer",
  "description": "Technical content creation for developer audiences",
  "category": "content",
  "icon": "pencil",
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "You are a developer relations content writer...",
  "temperature": 0.7,
  "max_steps": 12,
  "max_tokens": 4096,
  "allowed_skills": ["content_writer", "code_formatter", "seo_optimizer"],
  "require_approval": false,
  "risk_threshold": null,
  "is_builtin": false,
  "is_active": true,
  "tags": ["devrel", "content", "writing"],
  "metadata": {},
  "created_at": "2026-05-20T12:00:00Z",
  "created_by": "user-123"
}
```

### Delete Archetype

**`DELETE /api/v1/archetypes/{archetype_id}`** | Role: `operator`

```bash
curl -X DELETE https://your-instance.com/api/v1/archetypes/d4e5f6a7-b8c9-0123-defa-234567890123 \
  -H "Authorization: Bearer $TOKEN"
```

### Deploy Archetype

**`POST /api/v1/archetypes/{archetype_id}/deploy`** | Role: `operator`

Creates a new agent from an archetype template with optional overrides.

```bash
curl -X POST https://your-instance.com/api/v1/archetypes/d4e5f6a7-b8c9-0123-defa-234567890123/deploy \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Seed Built-in Archetypes

**`POST /api/v1/archetypes/seed`** | Role: `admin`

```bash
curl -X POST https://your-instance.com/api/v1/archetypes/seed \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "seeded": 8,
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Providers

Configure LLM provider API keys and manage model availability. Aria Core supports OpenAI, Anthropic, and xAI out of the box.

### List Configured Providers

**`GET /api/v1/providers`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/providers \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
[
  {
    "provider": "openai",
    "enabled": true,
    "default_model": "gpt-4o",
    "has_key": true,
    "key_preview": "sk-proj-...",
    "base_url": null
  },
  {
    "provider": "anthropic",
    "enabled": true,
    "default_model": "claude-sonnet-4-20250514",
    "has_key": true,
    "key_preview": "sk-ant-a...",
    "base_url": null
  }
]
```

### Configure Provider

**`POST /api/v1/providers`** | Role: `admin`

```bash
curl -X POST https://your-instance.com/api/v1/providers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "api_key": "sk-proj-your-key-here",
    "default_model": "gpt-4o",
    "enabled": true
  }'
```

**Response:**

```json
{
  "provider": "openai",
  "enabled": true,
  "default_model": "gpt-4o",
  "configured": true
}
```

### Remove Provider

**`DELETE /api/v1/providers/{provider}`** | Role: `admin`

```bash
curl -X DELETE https://your-instance.com/api/v1/providers/openai \
  -H "Authorization: Bearer $TOKEN"
```

### List Available Models

**`GET /api/v1/providers/models`** | Role: `auth`

```bash
# All models in registry
curl https://your-instance.com/api/v1/providers/models \
  -H "Authorization: Bearer $TOKEN"

# Filter by provider, only available (key configured)
curl "https://your-instance.com/api/v1/providers/models?provider=anthropic&available_only=true" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | null | Filter by provider: `openai`, `anthropic`, `xai` |
| `available_only` | bool | false | Only models with configured API keys |

### Provider Status

**`GET /api/v1/providers/status`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/providers/status \
  -H "Authorization: Bearer $TOKEN"
```

### Test Provider Connection

**`POST /api/v1/providers/{provider}/test`** | Role: `operator`

```bash
curl -X POST https://your-instance.com/api/v1/providers/openai/test \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "provider": "openai",
  "status": "connected",
  "adapter": "OpenAIAdapter"
}
```

---

## Events

Append-only event store with full replay capability for audit trails and state reconstruction.

### List Events

**`GET /api/v1/events`** | Role: `auth`

```bash
# All events
curl https://your-instance.com/api/v1/events \
  -H "Authorization: Bearer $TOKEN"

# Filter by type and agent
curl "https://your-instance.com/api/v1/events?event_type=agent.start&agent_id=c3d4e5f6-...&limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `event_type` | string | null | Filter by event type (e.g., `agent.start`, `plan.completed`) |
| `agent_id` | UUID | null | Filter by agent |
| `limit` | int | 100 | Max results (1-1000) |
| `offset` | int | 0 | Pagination offset |

**Response:**

```json
[
  {
    "id": "evt-001",
    "event_type": "agent.start",
    "agent_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "payload": {
      "model": "gpt-4o",
      "message": "Analyze Q2 revenue trends"
    },
    "timestamp": "2026-05-20T12:00:00Z"
  }
]
```

### Replay Events

**`GET /api/v1/events/replay`** | Role: `auth`

Replays events in chronological order for state reconstruction.

```bash
curl "https://your-instance.com/api/v1/events/replay?event_type=plan.completed&limit=5000" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `event_type` | string | null | Filter by event type |
| `agent_id` | UUID | null | Filter by agent |
| `limit` | int | 10000 | Max results (1-100000) |

**Response:**

```json
{
  "count": 142,
  "events": [
    {
      "event_type": "agent.start",
      "payload": { "..." : "..." },
      "timestamp": "2026-05-20T12:00:00Z"
    }
  ]
}
```

### Count Events

**`GET /api/v1/events/count`** | Role: `auth`

```bash
curl "https://your-instance.com/api/v1/events/count?event_type=agent.start" \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "count": 1247,
  "event_type": "agent.start"
}
```

---

## Contexts

Agent execution contexts capture the full state of a conversation or agent run.

### List Contexts

**`GET /api/v1/contexts`** | Role: `auth`

```bash
curl "https://your-instance.com/api/v1/contexts?limit=20" \
  -H "Authorization: Bearer $TOKEN"

# Filter by conversation
curl "https://your-instance.com/api/v1/contexts?conversation_id=e5f6a7b8-c9d0-1234-efab-345678901234" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `conversation_id` | UUID | null | Filter by conversation |
| `limit` | int | 50 | Max results (1-200) |
| `offset` | int | 0 | Pagination offset |

### Get Context

**`GET /api/v1/contexts/{context_id}`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/contexts/e5f6a7b8-c9d0-1234-efab-345678901234 \
  -H "Authorization: Bearer $TOKEN"
```

---

## Execution

Run an agent against a message with full FSM lifecycle, provider routing, time-travel checkpointing, and usage metering.

### Execute Agent

**`POST /api/v1/execute`** | Role: `auth`

```bash
curl -X POST https://your-instance.com/api/v1/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze the competitive landscape for AI agent frameworks",
    "agent_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "model": "gpt-4o",
    "conversation_id": "e5f6a7b8-c9d0-1234-efab-345678901234",
    "stream": false
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The input message to process |
| `agent_id` | UUID string | No | Registered agent to use (loads config) |
| `model` | string | No | Model override (takes priority over agent config) |
| `conversation_id` | UUID string | No | Continue an existing conversation |
| `stream` | bool | No | Enable streaming (default: false) |

**Model Resolution Priority:**

1. `model` field in request (explicit override)
2. Agent's configured model (from agent registry)
3. Default: `gpt-4`

**Response:**

```json
{
  "execution_id": "f6a7b8c9-d0e1-2345-fabc-456789012345",
  "agent_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "model_used": "gpt-4o",
  "response": "The AI agent framework landscape is rapidly evolving...",
  "state": "complete",
  "steps": 3,
  "duration_ms": 2847,
  "events": [
    {
      "event_type": "state.transition",
      "payload": {"from": "idle", "to": "routing"}
    },
    {
      "event_type": "state.transition",
      "payload": {"from": "routing", "to": "responding"}
    },
    {
      "event_type": "state.transition",
      "payload": {"from": "responding", "to": "complete"}
    }
  ],
  "checkpoints": 3,
  "error": null
}
```

**Error Response (provider not configured):**

```json
{
  "execution_id": "...",
  "agent_id": null,
  "model_used": "gpt-4o",
  "response": "",
  "state": "error",
  "steps": 0,
  "duration_ms": 12,
  "events": [],
  "checkpoints": 0,
  "error": "No API key configured for provider 'openai'. Configure it via POST /api/v1/providers with your API key."
}
```

---

## WebSocket

Real-time event streaming over WebSocket, scoped to the authenticated tenant.

### Connection Endpoint

**`WS /ws/events`**

### Connection Flow

1. Client opens WebSocket connection to `/ws/events`
2. Server accepts the connection
3. Client sends an authentication message (JSON)
4. Server validates the JWT and registers the connection for the tenant
5. Server sends a `connected` confirmation event
6. Server streams all tenant events in real-time
7. Client keeps alive by sending any text message (ping/pong)

### Authentication Message

```json
{
  "token": "Bearer eyJhbGciOiJIUzI1NiIs..."
}
```

### Connection Confirmation

```json
{
  "event_type": "connected",
  "payload": {
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-123",
    "connections": 1
  }
}
```

### Event Format

All events streamed over WebSocket follow this format:

```json
{
  "event_type": "agent.start",
  "payload": {
    "agent_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "model": "gpt-4o",
    "message": "Analyze this dataset"
  },
  "timestamp": "2026-05-20T12:00:00.000000Z"
}
```

### Common Event Types

| Event Type | Description |
|------------|-------------|
| `connected` | Connection established |
| `agent.start` | Agent execution started |
| `agent.complete` | Agent execution completed |
| `state.transition` | FSM state change |
| `plan.created` | Plan created |
| `plan.started` | Plan execution started |
| `plan.completed` | Plan execution completed |
| `action.started` | Plan action started |
| `action.completed` | Plan action completed |
| `approval.required` | Approval gate triggered |
| `approval.resolved` | Approval approved/rejected |

### Connection Status

**`GET /api/v1/ws/status`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/ws/status \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "tenant_connections": 3,
  "total_connections": 47
}
```

### JavaScript Client Example

```javascript
const ws = new WebSocket("wss://your-instance.com/ws/events");

ws.onopen = () => {
  ws.send(JSON.stringify({ token: "Bearer " + jwtToken }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.event_type}]`, data.payload);
};

ws.onerror = (error) => console.error("WebSocket error:", error);
ws.onclose = (event) => console.log("Disconnected:", event.code, event.reason);
```

---

## Billing

Usage metering and consumption tracking per tenant.

### Meter Types

| Meter | Description |
|-------|-------------|
| `api_call` | REST API requests |
| `event` | Events emitted to the event store |
| `agent_run` | Agent state machine executions |
| `plan_execution` | Plans executed |
| `ws_message` | WebSocket messages sent |
| `storage_bytes` | Approximate data stored |

### Get Usage Report

**`GET /api/v1/billing/usage`** | Role: `auth`

```bash
curl https://your-instance.com/api/v1/billing/usage \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "period_start": "2026-05-20T00:00:00Z",
  "period_end": "2026-05-20T14:32:00Z",
  "totals": {
    "api_call": 847,
    "event": 3291,
    "agent_run": 42,
    "plan_execution": 8
  },
  "record_count": 4188
}
```

### Get All Usage (Admin)

**`GET /api/v1/billing/usage/all`** | Role: `admin`

```bash
curl https://your-instance.com/api/v1/billing/usage/all \
  -H "Authorization: Bearer $TOKEN"
```

**Response:** Array of usage report objects for all tenants.

---

## Pricing

Public endpoints (no authentication required) for pricing tier information and cost estimation.

### Get Pricing Tiers

**`GET /api/v1/pricing/tiers`** | No auth

```bash
curl https://your-instance.com/api/v1/pricing/tiers
```

**Response:**

```json
[
  {
    "name": "Starter",
    "slug": "starter",
    "monthly_price": 0,
    "max_tenants": 1,
    "max_api_calls": 1000,
    "max_events": 5000,
    "max_agent_runs": 100,
    "max_agents": 5,
    "max_storage_gb": 1.0,
    "support_level": "Community",
    "features": [
      "Single tenant",
      "FSM runtime",
      "Risk scoring",
      "In-memory persistence",
      "REST API",
      "Config Portal"
    ],
    "is_custom": false
  },
  {
    "name": "Pro",
    "slug": "pro",
    "monthly_price": 99,
    "max_tenants": 5,
    "max_api_calls": 50000,
    "max_events": 250000,
    "max_agent_runs": 5000,
    "max_agents": 25,
    "max_storage_gb": 10.0,
    "support_level": "Email",
    "features": [
      "Multi-tenant (5)",
      "PostgreSQL persistence",
      "Event sourcing + replay",
      "Deep Bridge consensus",
      "WebSocket streaming",
      "JWT auth + RBAC",
      "Agent archetypes",
      "Usage billing"
    ],
    "is_custom": false
  },
  {
    "name": "Business",
    "slug": "business",
    "monthly_price": 499,
    "max_tenants": 25,
    "max_api_calls": 500000,
    "max_events": 2500000,
    "max_agent_runs": 50000,
    "max_agents": 100,
    "max_storage_gb": 100.0,
    "support_level": "Priority",
    "features": [
      "Multi-tenant (25)",
      "RS256 JWKS key rotation",
      "Helm chart deployment",
      "HPA autoscaling",
      "Custom risk policies per tenant",
      "Approval gate builder",
      "Stripe billing integration",
      "99.9% SLA"
    ],
    "is_custom": false
  },
  {
    "name": "Enterprise",
    "slug": "enterprise",
    "monthly_price": 0,
    "max_tenants": 999999,
    "max_api_calls": 999999999,
    "max_events": 999999999,
    "max_agent_runs": 999999999,
    "max_agents": 999999,
    "max_storage_gb": 999999.0,
    "support_level": "Dedicated",
    "features": [
      "Unlimited tenants",
      "White-label branding",
      "Custom domain",
      "Dedicated infrastructure",
      "SOC 2 compliance",
      "Custom SLA",
      "Onboarding + training",
      "24/7 dedicated support"
    ],
    "is_custom": true
  }
]
```

### Pricing Calculator

**`GET /api/v1/pricing/calculate`** | No auth

```bash
curl "https://your-instance.com/api/v1/pricing/calculate?api_calls=75000&events=300000&agent_runs=8000&agents=30&tenants=3&storage_gb=15"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `api_calls` | int | 0 | Projected monthly API calls |
| `events` | int | 0 | Projected monthly events |
| `agent_runs` | int | 0 | Projected monthly agent runs |
| `agents` | int | 0 | Number of agents |
| `tenants` | int | 1 | Number of tenants |
| `storage_gb` | float | 0 | Storage in GB |

**Response:**

```json
{
  "recommended_tier": "Business",
  "monthly_cost": 499.00,
  "base_cost": 499.00,
  "overage_cost": 0,
  "breakdown": {},
  "all_tiers": [
    {"tier": "Starter", "total_cost": 0, "recommended": false},
    {"tier": "Pro", "total_cost": 184.00, "recommended": false},
    {"tier": "Business", "total_cost": 499.00, "recommended": true},
    {"tier": "Enterprise", "total_cost": 0, "recommended": false}
  ]
}
```

### Overage Rates

| Resource | Rate |
|----------|------|
| API calls | $0.001/call over limit |
| Events | $0.0005/event over limit |
| Agent runs | $0.01/run over limit |
| Storage | $0.10/GB/month over limit |

---

## Health

Public health check endpoints for load balancers and orchestrators.

### Liveness Probe

**`GET /health`** | No auth

```bash
curl https://your-instance.com/health
```

### Readiness Probe

**`GET /ready`** | No auth

```bash
curl https://your-instance.com/ready
```

---

## Scheduling

> **Coming in v4.** Scheduled agent execution with cron expressions, recurring plans, and time-based triggers. Track progress at [ARIA-324].

---

## Error Codes

| Status | Code | Description | Resolution |
|--------|------|-------------|------------|
| `401` | Unauthorized | Missing, expired, or invalid JWT token | Check `Authorization` header format, verify token is not expired, ensure `ARIA_JWT_SECRET` matches |
| `403` | Forbidden | Insufficient role permissions | Your token's `role` claim does not meet the endpoint's minimum role requirement. Check the role column in the endpoint table above |
| `404` | Not Found | Resource does not exist or belongs to another tenant | Verify the UUID is correct. Cross-tenant access is denied by design -- only admins can see other tenants |
| `422` | Validation Error | Invalid request body or query parameters | Check the request body against the schema. FastAPI returns detailed field-level errors |
| `429` | Rate Limited | Too many requests within the sliding window | Respect `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers. Default: 60 requests/minute per tenant |
| `500` | Internal Error | Unexpected server error | Check server logs. Common causes: missing `python-jose` dependency, database connection failure |

### Rate Limit Headers

All authenticated `/api/*` responses include rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1716220800
```

### Security Headers

All responses include security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 0
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

---

## Complete Route Table

| # | Method | Path | Auth | Role | Description |
|---|--------|------|------|------|-------------|
| 1 | `GET` | `/health` | No | - | Liveness probe |
| 2 | `GET` | `/ready` | No | - | Readiness probe |
| 3 | `GET` | `/.well-known/jwks.json` | No | - | RS256 public keys |
| 4 | `GET` | `/api/v1/pricing/tiers` | No | - | Pricing tiers |
| 5 | `GET` | `/api/v1/pricing/calculate` | No | - | Cost calculator |
| 6 | `POST` | `/api/v1/tenants` | Yes | admin | Create tenant |
| 7 | `GET` | `/api/v1/tenants` | Yes | admin | List tenants |
| 8 | `GET` | `/api/v1/tenants/{id}` | Yes | auth | Get tenant |
| 9 | `PUT` | `/api/v1/tenants/{id}/config` | Yes | admin | Update config |
| 10 | `POST` | `/api/v1/plans` | Yes | operator | Create plan |
| 11 | `GET` | `/api/v1/plans` | Yes | auth | List plans |
| 12 | `GET` | `/api/v1/plans/{id}` | Yes | auth | Get plan |
| 13 | `DELETE` | `/api/v1/plans/{id}` | Yes | operator | Delete plan |
| 14 | `GET` | `/api/v1/approvals` | Yes | auth | List approvals |
| 15 | `GET` | `/api/v1/approvals/{id}` | Yes | auth | Get approval |
| 16 | `POST` | `/api/v1/approvals/{id}/approve` | Yes | operator | Approve |
| 17 | `POST` | `/api/v1/approvals/{id}/reject` | Yes | operator | Reject |
| 18 | `GET` | `/api/v1/agents` | Yes | auth | List agents |
| 19 | `POST` | `/api/v1/agents` | Yes | operator | Register agent |
| 20 | `GET` | `/api/v1/agents/{id}` | Yes | auth | Get agent |
| 21 | `DELETE` | `/api/v1/agents/{id}` | Yes | operator | Delete agent |
| 22 | `GET` | `/api/v1/providers` | Yes | auth | List providers |
| 23 | `POST` | `/api/v1/providers` | Yes | admin | Configure provider |
| 24 | `DELETE` | `/api/v1/providers/{provider}` | Yes | admin | Remove provider |
| 25 | `GET` | `/api/v1/providers/models` | Yes | auth | List models |
| 26 | `GET` | `/api/v1/providers/status` | Yes | auth | Provider status |
| 27 | `POST` | `/api/v1/providers/{provider}/test` | Yes | operator | Test connection |
| 28 | `GET` | `/api/v1/archetypes` | Yes | auth | List archetypes |
| 29 | `GET` | `/api/v1/archetypes/{id}` | Yes | auth | Get archetype |
| 30 | `POST` | `/api/v1/archetypes` | Yes | operator | Create archetype |
| 31 | `DELETE` | `/api/v1/archetypes/{id}` | Yes | operator | Delete archetype |
| 32 | `POST` | `/api/v1/archetypes/{id}/deploy` | Yes | operator | Deploy archetype |
| 33 | `POST` | `/api/v1/archetypes/seed` | Yes | admin | Seed defaults |
| 34 | `POST` | `/api/v1/execute` | Yes | auth | Execute agent |
| 35 | `GET` | `/api/v1/events` | Yes | auth | List events |
| 36 | `GET` | `/api/v1/events/replay` | Yes | auth | Replay events |
| 37 | `GET` | `/api/v1/events/count` | Yes | auth | Count events |
| 38 | `GET` | `/api/v1/contexts` | Yes | auth | List contexts |
| 39 | `GET` | `/api/v1/contexts/{id}` | Yes | auth | Get context |
| 40 | `GET` | `/api/v1/billing/usage` | Yes | auth | Usage report |
| 41 | `GET` | `/api/v1/billing/usage/all` | Yes | admin | All usage |
| 42 | `GET` | `/api/v1/ws/status` | Yes | auth | WS connection status |
| 43 | `WS` | `/ws/events` | JWT | - | Real-time stream |

---

Built by [Hidden Leaf Networks](https://hiddenleafnetworks.com) -- an applied AI studio.

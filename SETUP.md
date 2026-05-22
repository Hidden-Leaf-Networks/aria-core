# Aria Core — Setup Guide

Get Aria Core running locally or in production in under 5 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+ (for portal)
- Docker (optional, for PostgreSQL + production deploy)

## Quick Start (Development)

### 1. Install

```bash
git clone https://github.com/Hidden-Leaf-Networks/aria-core.git
cd aria-core

# Install with all optional deps
pip install -e ".[all,dev]"

# Or minimal (core only)
pip install -e "."
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
ARIA_JWT_SECRET=your-secret-here    # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"
ARIA_PERSISTENCE=memory              # memory for dev, postgres for production
ARIA_PORT=8100
```

### 3. Start the API

```bash
# Development (in-memory, auto-reload)
ARIA_JWT_SECRET=dev-secret uvicorn aria_core.api:create_app --factory --port 8100 --reload
```

Verify: `curl http://localhost:8100/health`

### 4. Start the Portal

```bash
cd portal
npm install
npm run dev
```

Opens at `http://localhost:3000` — proxies API calls to port 8100.

### 5. Get a JWT Token

```python
python -c "
from aria_core.api.auth import create_token, Role
from uuid import UUID
print(create_token(
    'admin',
    UUID('00000000-0000-0000-0000-000000000000'),
    Role.ADMIN,
    secret='dev-secret',
    tenant_slug='default',
    expires_in_seconds=86400
))
"
```

Paste the token into the portal login page.

## Configure Model Providers

After logging in, go to **Providers** and add your API keys:

| Provider | Key Format | Get Key |
|----------|-----------|---------|
| OpenAI | `sk-...` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | `sk-ant-...` | [console.anthropic.com](https://console.anthropic.com) |
| xAI | `xai-...` | [console.x.ai](https://console.x.ai) |

Or via API:

```bash
curl -X POST http://localhost:8100/api/v1/providers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","api_key":"sk-..."}'
```

## Execute Your First Agent

```bash
curl -X POST http://localhost:8100/api/v1/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Aria Core?","model":"gpt-4o-mini"}'
```

## Production Deploy (Docker)

### Option A: Docker Compose

```bash
cp .env.example .env
# Edit .env with production secrets

docker compose up -d
```

This starts:
- **API** on port 8000
- **PostgreSQL 16** on port 5432
- **Alembic migration** (runs once)

### Option B: Docker Compose + Portal + Caddy

```bash
# Build portal
cd portal && npm install && npm run build && cd ..

# Start everything
docker compose up -d

# Run Caddy for reverse proxy + SSL
docker run -d --name caddy \
  --network aria-core_default \
  -p 80:80 -p 443:443 \
  -v $(pwd)/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v $(pwd)/portal/dist:/srv/portal:ro \
  caddy:2-alpine
```

### Option C: Tailscale Funnel (no port forwarding needed)

```bash
# On the server
tailscale funnel --bg 8200
```

Gives you a public HTTPS URL instantly.

## Production Deploy (Kubernetes)

```bash
# Create secrets
kubectl create secret generic aria-core-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=postgres-password=$(openssl rand -base64 16)

# Install
helm install aria-core deploy/helm/aria-core \
  --set ingress.hosts[0].host=aria.yourdomain.com
```

## Project Structure

```
aria-core/
├── src/aria_core/          # Python backend (32 modules)
│   ├── api/                # FastAPI REST API (40+ routes)
│   ├── runtime/            # FSM state machine, time-travel
│   ├── adapters/           # LLM adapters (OpenAI, Anthropic, xAI)
│   ├── providers/          # Model registry + adapter factory
│   ├── tenant/             # Multi-tenant isolation
│   ├── persistence/        # InMemory + PostgreSQL providers
│   ├── permissions/        # Risk scoring + approval gates
│   ├── orchestration/      # Deep Bridge multi-model consensus
│   ├── planning/           # Plan engine with dependencies
│   ├── archetypes/         # 14 agent templates + marketplace
│   ├── mcp/                # MCP server + client + recursive
│   ├── a2a/                # A2A protocol server + client
│   ├── flows/              # Flow orchestration (DAG)
│   ├── memory/             # Cross-session agent memory
│   ├── eval/               # Eval framework + production metrics
│   ├── scheduler/          # Cron + interval scheduling
│   ├── knowledge/          # RAG with vector search
│   ├── voice/              # Deepgram STT + ElevenLabs TTS
│   ├── avatar/             # 3D avatar with lip sync
│   ├── platform/           # Multi-platform presence
│   ├── phone/              # Phone agent (Plivo/Twilio)
│   ├── computer_use/       # Desktop automation
│   ├── billing/            # Stripe metered billing
│   ├── training/           # Agent guardrails
│   ├── copilot/            # AI config builder
│   ├── collaboration/      # Real-time multiplayer
│   ├── deploy/             # GitHub deploy
│   ├── auth/               # Fine-grained capabilities
│   ├── protocols/          # W3C compatibility
│   ├── telemetry/          # OpenTelemetry tracing
│   ├── a2ui/               # Generative UI
│   └── data_cloud/         # Query federation
├── portal/                 # React 19 + Vite + Tailwind portal
│   └── src/pages/          # 12 pages (Dashboard, Tenants, Plans, etc.)
├── deploy/helm/            # Kubernetes Helm chart
├── tests/                  # 1,056 tests
├── docker-compose.yml      # Docker production config
├── Dockerfile              # Multi-stage build
└── alembic.ini             # Database migrations
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/unit/test_runtime.py -v

# With coverage
python -m pytest tests/ --cov=aria_core --cov-report=html
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARIA_JWT_SECRET` | *required* | JWT signing secret |
| `ARIA_PERSISTENCE` | `memory` | `memory` or `postgres` |
| `ARIA_DATABASE_URL` | `postgresql+asyncpg://aria:aria@localhost:5432/aria_core` | PostgreSQL connection |
| `ARIA_PORT` | `8000` | API server port |
| `ARIA_DEBUG` | `false` | Enable debug logging |
| `ARIA_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `ARIA_JWT_ALGORITHM` | `HS256` | JWT algorithm (HS256 or RS256) |

## Live Instance

- **Public URL:** https://axis-node00.tail669549.ts.net
- **API Health:** https://axis-node00.tail669549.ts.net/health

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

Built by [Hidden Leaf Networks](https://hiddenleafnetworks.com)

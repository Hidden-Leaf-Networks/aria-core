# Aria Core Landing Page Content

> Content source for `aria.hiddenleafnetworks.com`. Each section maps to a React component for the Next.js build.

---

## 1. Hero

> **Component:** `<HeroSection />`

**Headline:**
Build AI Agent Platforms That Actually Go to Production

**Subheadline:**
Aria Core is the open-source multi-tenant agent framework with deterministic execution, permission-first safety, and white-label SaaS built in. Stop prototyping. Start shipping.

**CTA Button:**
Get Started Free

**Secondary CTA:**
View Documentation

**Background note:** Dark gradient with subtle grid pattern. The hero should convey precision and control, not chatbot energy.

---

## 2. Problem Statement

> **Component:** `<ProblemSection />`

**Section headline:**
Agent Frameworks Were Built for Demos, Not Production

**Body copy:**

Every team building AI agents hits the same wall. The prototype works. The demo impresses. Then reality arrives.

- **No guardrails.** Agents loop forever, hallucinate actions, and execute without oversight. One bad prompt costs you a customer.
- **No multi-tenancy.** You need to serve multiple clients from one platform, but your framework assumes a single user with a single API key.
- **No audit trail.** When something goes wrong at 2 AM, you have no idea what the agent did, why it did it, or how to replay the failure.
- **No path to SaaS.** You want to sell your agent platform, but you are stuck building auth, billing, tenant isolation, and admin portals from scratch.

Aria Core exists because we hit every one of these walls while running 14 production agents. We extracted the solution.

---

## 3. Feature Grid

> **Component:** `<FeatureGrid />` -- 2x4 grid on desktop, single column on mobile. Each card has an icon, title, and 2-sentence description.

### Feature 1: Deterministic FSM Runtime
**Icon suggestion:** `CircuitBoard` or `Workflow`

Every agent runs through an 8-state finite state machine with validated transitions. No uncontrolled loops, no infinite recursion, no mystery. Max-step enforcement kills runaway agents before they burn your budget.

### Feature 2: Multi-Model Consensus
**Icon suggestion:** `GitMerge` or `Vote`

Deep Bridge queries multiple LLMs in parallel and synthesizes their responses for high-stakes decisions. When the answer matters, one model is not enough. Supports OpenAI, Anthropic, and xAI out of the box.

### Feature 3: Permission-First Safety
**Icon suggestion:** `ShieldCheck` or `Lock`

Every action is risk-scored from 0-100. High-risk actions trigger approval gates with full RBAC. Immutable decision records create a compliance-ready audit trail. Your agents do not act without permission.

### Feature 4: Multi-Tenant Isolation
**Icon suggestion:** `Building2` or `Layers`

Tenant-scoped persistence, config overrides, rate limiting, and billing. Each tenant is a walled garden. No data leaks, no cross-contamination, no shared state. Built for SaaS from day one.

### Feature 5: Event Sourcing + Replay
**Icon suggestion:** `History` or `RotateCcw`

Append-only event store captures every state transition, every decision, every action. Full replay capability lets you reconstruct any execution. Debug production issues by rewinding time.

### Feature 6: White-Label Ready
**Icon suggestion:** `Palette` or `Paintbrush`

Config Portal with per-tenant branding, custom domains, and theme overrides. JWT auth with RBAC. Your clients see your brand, not ours. Ship a SaaS product, not a framework demo.

### Feature 7: Agent Archetypes
**Icon suggestion:** `Copy` or `LayoutTemplate`

Pre-built agent templates for research, engineering, content, data, support, security, and operations. Deploy a production-ready agent in one API call. Customize with overrides or build your own.

### Feature 8: Usage Billing
**Icon suggestion:** `Receipt` or `CreditCard`

Built-in usage metering tracks API calls, agent runs, events, and storage per tenant. Stripe integration for automated billing. Four pricing tiers from free to enterprise, with usage-based overages.

---

## 4. How It Works

> **Component:** `<HowItWorks />` -- 3-step horizontal flow with numbered circles and connecting lines.

### Step 1: Configure
**Icon suggestion:** `Settings` or `Sliders`

Set up your tenant, configure LLM providers with your API keys, and deploy agent archetypes. Define risk policies and approval gates. Five minutes from `pip install` to running API.

```bash
pip install aria-core[all]
ARIA_JWT_SECRET=your-secret uvicorn aria_core.api:create_app --factory
```

### Step 2: Deploy
**Icon suggestion:** `Rocket` or `Upload`

Register agents, create execution plans with dependency graphs, and start processing messages through the deterministic FSM. Every action is risk-scored, every decision is logged.

```bash
curl -X POST https://your-instance.com/api/v1/execute \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message": "Analyze competitor pricing", "model": "gpt-4o"}'
```

### Step 3: Scale
**Icon suggestion:** `TrendingUp` or `Maximize`

Add tenants, onboard clients, monitor usage through billing dashboards. Deploy with Docker, Helm, or HPA autoscaling. The framework scales with you from solo developer to multi-tenant SaaS.

```bash
docker-compose up  # or helm install aria-core ./deploy/helm
```

---

## 5. Competitive Comparison

> **Component:** `<ComparisonTable />` -- Responsive table with sticky first column. Checkmarks and X marks for boolean features, text for descriptive ones.

| Feature | Aria Core | LangGraph | CrewAI |
|---------|-----------|-----------|--------|
| **Deterministic FSM** | 8-state validated machine | No (graph-based, no enforcement) | No (sequential/hierarchical) |
| **Multi-model consensus** | Deep Bridge parallel voting | Basic model routing | None |
| **Risk scoring** | 0-100 per action with approval gates | None | None |
| **Multi-tenant** | Full isolation with config overrides | No | No |
| **White-label** | Config Portal + branding + custom domains | No | No |
| **Event sourcing** | Append-only with full replay | Checkpoints only | None |
| **Usage billing** | Stripe metered billing built-in | Pricing page (no metering) | None |
| **RBAC** | Admin / Operator / Viewer hierarchy | None | None |
| **Agent templates** | Archetype registry with one-click deploy | None | Role-based (limited) |
| **Deployment** | Docker + Helm + HPA autoscaling | Cloud-hosted only | Docker |
| **WebSocket streaming** | Tenant-scoped real-time events | LangSmith (separate) | None |
| **License** | Apache 2.0 | MIT | MIT |
| **Production heritage** | Extracted from 14-agent production system | Research-oriented | Crew simulation |

---

## 6. Pricing

> **Component:** `<PricingSection />` -- 4 cards in a row, Business tier highlighted as "Most Popular". Enterprise card has "Contact Us" instead of a price.

### Starter -- Free

- 1 tenant
- 5 agents
- 1,000 API calls/month
- 100 agent runs/month
- 5,000 events/month
- 1 GB storage
- In-memory persistence
- FSM runtime + risk scoring
- REST API + Config Portal
- Community support

**CTA:** Get Started Free

### Pro -- $99/month

- 5 tenants
- 25 agents
- 50,000 API calls/month
- 5,000 agent runs/month
- 250,000 events/month
- 10 GB storage
- PostgreSQL persistence
- Event sourcing + replay
- Deep Bridge consensus
- WebSocket streaming
- JWT auth + RBAC
- Agent archetypes
- Usage billing
- Email support

**CTA:** Start Pro Trial

### Business -- $499/month

*Most Popular*

- 25 tenants
- 100 agents
- 500,000 API calls/month
- 50,000 agent runs/month
- 2,500,000 events/month
- 100 GB storage
- RS256 JWKS key rotation
- Helm chart deployment
- HPA autoscaling
- Custom risk policies per tenant
- Approval gate builder
- Stripe billing integration
- 99.9% SLA
- Priority support

**CTA:** Start Business Trial

### Enterprise -- Custom

- Unlimited tenants
- Unlimited agents
- Unlimited API calls
- Unlimited agent runs
- White-label branding
- Custom domain
- Dedicated infrastructure
- SOC 2 compliance
- Custom SLA
- Onboarding + training
- 24/7 dedicated support

**CTA:** Contact Sales

### Overage Pricing

Usage beyond tier limits is billed at:

| Resource | Rate |
|----------|------|
| API calls | $0.001/call |
| Events | $0.0005/event |
| Agent runs | $0.01/run |
| Storage | $0.10/GB/month |

---

## 7. Testimonial Placeholder

> **Component:** `<TestimonialSection />` -- 3 testimonial cards with avatar, name, title, company, and quote. Carousel on mobile.

### Slot 1
**Name:** [First Client Name]
**Title:** [Title]
**Company:** [Company]
**Avatar:** Placeholder circle with initials
**Quote:** "[Testimonial about production reliability and multi-tenant isolation]"

### Slot 2
**Name:** [Second Client Name]
**Title:** [Title]
**Company:** [Company]
**Avatar:** Placeholder circle with initials
**Quote:** "[Testimonial about speed of deployment and white-label capabilities]"

### Slot 3
**Name:** [Third Client Name]
**Title:** [Title]
**Company:** [Company]
**Avatar:** Placeholder circle with initials
**Quote:** "[Testimonial about safety features and approval gates preventing costly mistakes]"

> **Implementation note:** Hide this section until at least one testimonial is populated. Use a feature flag or conditional render.

---

## 8. CTA Section

> **Component:** `<CTASection />` -- Full-width dark background with centered text and two buttons.

**Headline:**
Ship Your Agent Platform This Week

**Subheadline:**
Aria Core is open-source under Apache 2.0. Start with the Starter tier for free, scale to Enterprise when you are ready.

**Primary CTA Button:**
Get Started Free

**Secondary CTA Button:**
Read the Docs

**Tertiary link:**
Star us on GitHub

---

## 9. FAQ

> **Component:** `<FAQSection />` -- Accordion-style expand/collapse. All collapsed by default.

### Q: Is Aria Core open source?

Yes. Aria Core is licensed under Apache 2.0. The full source code is available on GitHub. You can self-host, fork, and modify freely. The paid tiers cover managed hosting, premium support, and enterprise features like SOC 2 compliance and dedicated infrastructure.

### Q: What LLM providers does Aria Core support?

Aria Core ships with adapters for OpenAI, Anthropic, and xAI. The adapter interface is provider-agnostic, so you can add custom adapters for any LLM. Deep Bridge multi-model consensus works across all configured providers simultaneously.

### Q: How is Aria Core different from LangChain or LangGraph?

LangChain is a toolkit for building LLM applications. LangGraph adds graph-based workflows. Aria Core is an agent platform framework -- it includes the runtime, but also multi-tenancy, RBAC, billing, approval gates, event sourcing, and white-label deployment. If you are building a SaaS product powered by AI agents, Aria Core gives you the full platform, not just the execution engine.

### Q: Can I use Aria Core without the API server?

Yes. Aria Core works as a standalone Python library. Import the state machine, router, planner, and adapters directly into your application. The API server, auth, billing, and multi-tenancy layers are optional -- install only what you need with extras like `pip install aria-core[openai]`.

### Q: What does "deterministic FSM" mean in practice?

Every agent execution follows a fixed state machine: IDLE, ROUTING, PLANNING, EXECUTING, BLOCKED, RESPONDING, COMPLETE, or ERROR. Transitions between states are validated -- invalid transitions are rejected. A max-step limit kills runaway agents. This means you can reason about what your agent will do, audit what it did, and guarantee it will terminate. No infinite loops, no mystery behavior.

### Q: How does multi-tenant isolation work?

Every piece of data in Aria Core is scoped to a tenant ID. Persistence, events, agents, plans, configs, and billing are all tenant-isolated. A tenant context is injected at the API layer and validated on every database call. There is no way to access another tenant's data through the API. Rate limiting is also per-tenant, so one noisy client cannot degrade service for others.

---

## Component Mapping Summary

| Section | Component | React File (suggested) |
|---------|-----------|------------------------|
| Hero | `<HeroSection />` | `hero-section.tsx` |
| Problem | `<ProblemSection />` | `problem-section.tsx` |
| Features | `<FeatureGrid />` | `feature-grid.tsx` |
| How It Works | `<HowItWorks />` | `how-it-works.tsx` |
| Comparison | `<ComparisonTable />` | `comparison-table.tsx` |
| Pricing | `<PricingSection />` | `pricing-section.tsx` |
| Testimonials | `<TestimonialSection />` | `testimonial-section.tsx` |
| CTA | `<CTASection />` | `cta-section.tsx` |
| FAQ | `<FAQSection />` | `faq-section.tsx` |

---

## Design Notes

- **Color palette:** Inherit from HLN brand. Dark backgrounds with accent highlights for CTAs.
- **Typography:** Clean sans-serif. Headlines bold, body regular weight.
- **Code blocks:** Dark theme with syntax highlighting for `bash` and `json` examples.
- **Responsive:** Mobile-first. Feature grid collapses to single column. Pricing cards stack vertically. Comparison table scrolls horizontally.
- **Animations:** Subtle fade-in on scroll for each section. No heavy animations -- this is a developer tool, not a consumer app.

---

Built by [Hidden Leaf Networks](https://hiddenleafnetworks.com) -- an applied AI studio.

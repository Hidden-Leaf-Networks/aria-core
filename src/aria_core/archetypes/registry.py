"""Archetype registry — CRUD + built-in defaults."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.archetypes.models import Archetype, ArchetypeCategory


# Built-in archetypes shipped with aria-core
BUILTIN_ARCHETYPES: list[Archetype] = [
    Archetype(
        name="Research Analyst",
        slug="research-analyst",
        description="Deep research and analysis across multiple sources. Synthesizes findings into structured reports.",
        category=ArchetypeCategory.RESEARCH,
        icon="🔍",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a meticulous research analyst. Gather information from multiple sources, cross-reference findings, identify patterns, and produce clear, well-structured reports with citations.",
        temperature=0.3,
        max_steps=20,
        allowed_skills=["web_search", "doc_parse", "synthesize", "generate_report"],
        is_builtin=True,
        tags=["research", "analysis", "reports"],
    ),
    Archetype(
        name="Code Assistant",
        slug="code-assistant",
        description="Software development assistant for code generation, review, and debugging.",
        category=ArchetypeCategory.ENGINEERING,
        icon="💻",
        model="claude-sonnet-4-20250514",
        system_prompt="You are an expert software engineer. Write clean, well-tested code. Follow best practices for the target language. Explain your reasoning and suggest improvements.",
        temperature=0.2,
        max_steps=15,
        allowed_skills=["code_generate", "code_review", "test_generate", "refactor"],
        is_builtin=True,
        tags=["code", "engineering", "development"],
    ),
    Archetype(
        name="Content Writer",
        slug="content-writer",
        description="Professional content creation with brand voice consistency and SEO awareness.",
        category=ArchetypeCategory.CONTENT,
        icon="✍️",
        model="gpt-4o",
        system_prompt="You are a professional content writer. Create engaging, well-structured content that matches the brand voice. Optimize for readability and SEO. Adapt tone to the target audience.",
        temperature=0.8,
        max_steps=10,
        allowed_skills=["llm_generate", "seo_optimize", "tone_check", "publish"],
        is_builtin=True,
        tags=["content", "writing", "seo", "marketing"],
    ),
    Archetype(
        name="Data Engineer",
        slug="data-engineer",
        description="Data pipeline design, SQL generation, schema design, and ETL orchestration.",
        category=ArchetypeCategory.DATA,
        icon="📊",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a senior data engineer. Design efficient data pipelines, write optimized SQL, and build robust ETL processes. Prioritize data quality, idempotency, and observability.",
        temperature=0.1,
        max_steps=15,
        allowed_skills=["sql_generate", "schema_design", "extract", "transform", "load", "validate"],
        is_builtin=True,
        tags=["data", "sql", "etl", "pipelines"],
    ),
    Archetype(
        name="Customer Support Agent",
        slug="support-agent",
        description="Customer-facing support with empathy, knowledge base lookup, and escalation awareness.",
        category=ArchetypeCategory.SUPPORT,
        icon="💬",
        model="gpt-4o",
        system_prompt="You are a friendly, knowledgeable customer support agent. Listen carefully, provide accurate answers, and escalate when needed. Always maintain a helpful and empathetic tone.",
        temperature=0.5,
        max_steps=10,
        allowed_skills=["kb_search", "ticket_create", "escalate", "send_email"],
        is_builtin=True,
        tags=["support", "customer", "helpdesk"],
    ),
    Archetype(
        name="Security Auditor",
        slug="security-auditor",
        description="Security analysis, vulnerability assessment, and compliance checking.",
        category=ArchetypeCategory.SECURITY,
        icon="🛡️",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a security auditor. Analyze code, configurations, and infrastructure for vulnerabilities. Reference OWASP, CWE, and NIST standards. Provide actionable remediation steps with severity ratings.",
        temperature=0.1,
        max_steps=20,
        require_approval=True,
        risk_threshold=40,
        allowed_skills=["code_scan", "config_audit", "vuln_check", "compliance_check"],
        is_builtin=True,
        tags=["security", "audit", "compliance", "vulnerability"],
    ),
    Archetype(
        name="Ops Coordinator",
        slug="ops-coordinator",
        description="Operations orchestration — monitors, triages, and coordinates responses across systems.",
        category=ArchetypeCategory.OPERATIONS,
        icon="⚡",
        model="gpt-4",
        system_prompt="You are an operations coordinator. Monitor system health, triage incidents, and coordinate responses. Prioritize by severity. Communicate clearly and escalate when thresholds are breached.",
        temperature=0.3,
        max_steps=25,
        allowed_skills=["metrics_collect", "alert_triage", "runbook_execute", "notify", "escalate"],
        is_builtin=True,
        tags=["ops", "monitoring", "incident", "coordination"],
    ),
    Archetype(
        name="Multi-Platform Agent",
        slug="multi-platform-agent",
        description="Deploy agent presence across Chat, Discord, Slack, and X simultaneously. Manages multi-channel communication and context.",
        category=ArchetypeCategory.OPERATIONS,
        icon="📡",
        model="gpt-4o",
        system_prompt="You are a multi-platform communications agent. Maintain consistent presence across Chat, Discord, Slack, and X. Synchronize context between channels, identify users across platforms, and manage threaded conversations seamlessly.",
        temperature=0.5,
        max_steps=15,
        allowed_skills=["channel_send", "channel_listen", "context_sync", "user_identify", "thread_manage"],
        is_builtin=True,
        tags=["multi-platform", "chat", "discord", "slack", "integration"],
    ),
    Archetype(
        name="Phone Agent",
        slug="phone-agent",
        description="Voice-driven agent for phone calls — appointment booking, customer intake, outbound outreach with natural conversation.",
        category=ArchetypeCategory.SUPPORT,
        icon="📞",
        model="gpt-4o",
        system_prompt="You are a professional phone agent. Handle inbound and outbound calls with natural conversation flow. Book appointments, conduct customer intake, and perform outreach. Maintain a warm, clear tone and confirm details before ending calls.",
        temperature=0.6,
        max_steps=10,
        allowed_skills=["phone_call", "speech_to_text", "text_to_speech", "calendar_book", "contact_lookup"],
        is_builtin=True,
        tags=["voice", "phone", "calls", "booking", "outreach"],
    ),
    Archetype(
        name="Desktop Automation Agent",
        slug="desktop-automation",
        description="Computer-use agent for desktop automation — screen interaction, application control, data entry, and testing.",
        category=ArchetypeCategory.OPERATIONS,
        icon="🖥️",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a desktop automation agent. Interact with screen elements, control applications, perform data entry, and execute test scenarios. Always verify actions before proceeding and capture screenshots for confirmation.",
        temperature=0.2,
        max_steps=30,
        require_approval=True,
        risk_threshold=60,
        allowed_skills=["screen_capture", "mouse_click", "keyboard_type", "app_launch", "ocr_read", "window_manage"],
        is_builtin=True,
        tags=["desktop", "automation", "computer-use", "rpa", "testing"],
    ),
    Archetype(
        name="Web Scraping Agent",
        slug="web-scraper",
        description="High-speed web data collection with structured extraction, pagination handling, and rate limit awareness.",
        category=ArchetypeCategory.DATA,
        icon="🕷️",
        model="gpt-4o",
        system_prompt="You are a web scraping specialist. Collect data from websites efficiently with structured extraction. Handle pagination, respect rate limits, rotate proxies when needed, and output clean, validated data in the requested format.",
        temperature=0.1,
        max_steps=20,
        allowed_skills=["web_scrape", "html_parse", "pagination_follow", "data_extract", "rate_limit", "proxy_rotate"],
        is_builtin=True,
        tags=["scraping", "data-collection", "web", "extraction", "crawl"],
    ),
    Archetype(
        name="Site Cloner Agent",
        slug="site-cloner",
        description="Reverse-engineer any website into clean React/Next.js components. Extracts design tokens, layout, and interaction patterns.",
        category=ArchetypeCategory.ENGINEERING,
        icon="🔄",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a site cloning specialist. Analyze websites to extract design tokens, layout structures, and interaction patterns. Generate clean React/Next.js components with proper Tailwind CSS styling. Preserve visual fidelity while producing maintainable code.",
        temperature=0.3,
        max_steps=25,
        allowed_skills=["site_fetch", "css_extract", "component_generate", "design_token_extract", "layout_analyze"],
        is_builtin=True,
        tags=["clone", "react", "design-system", "frontend", "reverse-engineer"],
    ),
    Archetype(
        name="Video Analytics Agent",
        slug="video-analytics",
        description="Real-time video stream processing — object detection, scene classification, anomaly detection, and frame analysis.",
        category=ArchetypeCategory.DATA,
        icon="📹",
        model="gpt-4o",
        system_prompt="You are a video analytics agent. Process video streams in real-time to detect objects, classify scenes, flag anomalies, and analyze individual frames. Generate structured reports with timestamps, confidence scores, and visual annotations.",
        temperature=0.1,
        max_steps=15,
        allowed_skills=["video_ingest", "frame_extract", "object_detect", "scene_classify", "anomaly_flag", "report_generate"],
        is_builtin=True,
        tags=["video", "analytics", "vision", "detection", "streaming"],
    ),
    Archetype(
        name="Workflow Generator Agent",
        slug="workflow-generator",
        description="Generates automation workflows compatible with n8n, Zapier, and Make. Converts natural language to executable workflow JSON.",
        category=ArchetypeCategory.ENGINEERING,
        icon="⚙️",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a workflow automation specialist. Convert natural language descriptions into executable workflow definitions for n8n, Zapier, and Make. Configure triggers, actions, and conditional logic. Output valid, importable JSON for each platform.",
        temperature=0.3,
        max_steps=15,
        allowed_skills=["workflow_design", "n8n_export", "zapier_export", "make_export", "trigger_config", "action_chain"],
        is_builtin=True,
        tags=["workflow", "n8n", "zapier", "automation", "integration", "no-code"],
    ),
]


class ArchetypeRegistry:
    """Manages agent archetypes per tenant with built-in defaults."""

    def __init__(self) -> None:
        # {tenant_id: {archetype_id: Archetype}}
        self._store: dict[UUID, dict[UUID, Archetype]] = {}

    def _ensure_tenant(self, tenant_id: UUID) -> dict[UUID, Archetype]:
        if tenant_id not in self._store:
            self._store[tenant_id] = {}
        return self._store[tenant_id]

    async def seed_defaults(self, tenant_id: UUID) -> int:
        """Seed built-in archetypes for a tenant. Returns count seeded."""
        store = self._ensure_tenant(tenant_id)
        existing_slugs = {a.slug for a in store.values()}
        count = 0
        for builtin in BUILTIN_ARCHETYPES:
            if builtin.slug not in existing_slugs:
                copy = builtin.model_copy(update={"tenant_id": tenant_id})
                store[copy.id] = copy
                count += 1
        return count

    async def list(
        self,
        tenant_id: UUID,
        category: str | None = None,
        active_only: bool = True,
    ) -> list[Archetype]:
        """List archetypes for a tenant."""
        store = self._ensure_tenant(tenant_id)
        archetypes = list(store.values())
        if category:
            archetypes = [a for a in archetypes if a.category.value == category]
        if active_only:
            archetypes = [a for a in archetypes if a.is_active]
        return sorted(archetypes, key=lambda a: (not a.is_builtin, a.name))

    async def get(self, tenant_id: UUID, archetype_id: UUID) -> Archetype | None:
        return self._ensure_tenant(tenant_id).get(archetype_id)

    async def get_by_slug(self, tenant_id: UUID, slug: str) -> Archetype | None:
        for a in self._ensure_tenant(tenant_id).values():
            if a.slug == slug:
                return a
        return None

    async def save(self, tenant_id: UUID, archetype: Archetype) -> Archetype:
        store = self._ensure_tenant(tenant_id)
        archetype = archetype.model_copy(update={"tenant_id": tenant_id})
        store[archetype.id] = archetype
        return archetype

    async def delete(self, tenant_id: UUID, archetype_id: UUID) -> bool:
        store = self._ensure_tenant(tenant_id)
        if archetype_id in store:
            del store[archetype_id]
            return True
        return False

    async def create_from_archetype(
        self,
        tenant_id: UUID,
        archetype_id: UUID,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an agent config dict from an archetype with optional overrides."""
        archetype = await self.get(tenant_id, archetype_id)
        if not archetype:
            raise ValueError(f"Archetype {archetype_id} not found")

        config = {
            "name": archetype.name,
            "slug": archetype.slug,
            "description": archetype.description,
            "model": archetype.model,
            "system_prompt": archetype.system_prompt,
            "temperature": archetype.temperature,
            "max_steps": archetype.max_steps,
            "allowed_skills": list(archetype.allowed_skills),
        }
        if overrides:
            config.update(overrides)
        return config

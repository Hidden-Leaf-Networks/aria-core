import { useState, useMemo } from "react";
import { HudTopBar, HudPanel } from "@/components/hud";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Review {
  author: string;
  rating: number;
  comment: string;
  date: string;
}

interface ArchetypeListing {
  id: string;
  icon: string;
  name: string;
  slug: string;
  category: Category;
  publisher: string;
  rating: number;
  reviewCount: number;
  downloads: number;
  shortDescription: string;
  fullDescription: string;
  model: string;
  temperature: number;
  maxSteps: number;
  systemPromptPreview: string;
  reviews: Review[];
  createdAt: string;
}

type Category = "Research" | "Engineering" | "Content" | "Data" | "Support" | "Security" | "Operations";
type SortOption = "downloads" | "rating" | "newest";

const CATEGORIES: readonly ("All" | Category)[] = [
  "All", "Research", "Engineering", "Content", "Data", "Support", "Security", "Operations",
] as const;

const CATEGORY_COLORS: Record<Category, string> = {
  Research: "#3B82F6",
  Engineering: "#00FFAA",
  Content: "#F59E0B",
  Data: "#8B5CF6",
  Support: "#EC4899",
  Security: "#EF4444",
  Operations: "#06B6D4",
};

/* ------------------------------------------------------------------ */
/*  Mock Data                                                          */
/* ------------------------------------------------------------------ */

const MOCK_LISTINGS: ArchetypeListing[] = [
  {
    id: "mkt-001", icon: "🔍", name: "Deep Research Analyst", slug: "deep-research-analyst",
    category: "Research", publisher: "Hidden Leaf Networks", rating: 4.8, reviewCount: 127, downloads: 3420,
    shortDescription: "Multi-source research synthesis with structured report output.",
    fullDescription: "A meticulous research analyst that gathers information from multiple sources, cross-references findings, identifies patterns, and produces clear, well-structured reports with citations. Ideal for market research, competitive analysis, and literature reviews.",
    model: "claude-sonnet-4-20250514", temperature: 0.3, maxSteps: 20,
    systemPromptPreview: "You are a meticulous research analyst. Gather information from multiple sources, cross-reference findings, identify patterns...",
    reviews: [
      { author: "dataops_team", rating: 5, comment: "Best research agent we've deployed. Saves hours of manual work.", date: "2026-05-15" },
      { author: "mktg_lead", rating: 4, comment: "Great at synthesis but sometimes over-cites. Tuned temperature down to 0.2 for our use case.", date: "2026-05-10" },
    ],
    createdAt: "2026-04-01",
  },
  {
    id: "mkt-002", icon: "💻", name: "Fullstack Engineer", slug: "fullstack-engineer",
    category: "Engineering", publisher: "Hidden Leaf Networks", rating: 4.9, reviewCount: 215, downloads: 5810,
    shortDescription: "Production-grade code generation with testing and review.",
    fullDescription: "Expert software engineer specializing in full-stack development. Generates clean, well-tested code following best practices. Supports TypeScript, Python, Go, and Rust with framework-specific patterns for React, Next.js, FastAPI, and more.",
    model: "claude-sonnet-4-20250514", temperature: 0.2, maxSteps: 25,
    systemPromptPreview: "You are an expert full-stack software engineer. Write clean, well-tested, production-ready code...",
    reviews: [
      { author: "eng_director", rating: 5, comment: "Our team's productivity doubled. The code quality is consistently high.", date: "2026-05-18" },
      { author: "solo_dev", rating: 5, comment: "Like having a senior engineer on call 24/7.", date: "2026-05-12" },
    ],
    createdAt: "2026-03-15",
  },
  {
    id: "mkt-003", icon: "✍️", name: "Brand Voice Writer", slug: "brand-voice-writer",
    category: "Content", publisher: "ContentForge Labs", rating: 4.5, reviewCount: 89, downloads: 2150,
    shortDescription: "SEO-optimized content creation with brand voice consistency.",
    fullDescription: "Professional content writer that maintains brand voice consistency across all outputs. Produces blog posts, social media copy, email campaigns, and landing page content. Built-in SEO awareness and readability optimization.",
    model: "gpt-4o", temperature: 0.8, maxSteps: 10,
    systemPromptPreview: "You are a professional content writer. Create engaging, well-structured content that matches the brand voice...",
    reviews: [
      { author: "content_mgr", rating: 5, comment: "Finally, AI content that doesn't sound like AI.", date: "2026-05-14" },
      { author: "startup_cmo", rating: 4, comment: "Good baseline. We customized the system prompt for our tone.", date: "2026-05-08" },
    ],
    createdAt: "2026-04-10",
  },
  {
    id: "mkt-004", icon: "📊", name: "Pipeline Architect", slug: "pipeline-architect",
    category: "Data", publisher: "DataStack Co", rating: 4.7, reviewCount: 64, downloads: 1890,
    shortDescription: "Data pipeline design, SQL generation, and ETL orchestration.",
    fullDescription: "Senior data engineer that designs efficient data pipelines, writes optimized SQL, and builds robust ETL processes. Supports Snowflake, BigQuery, Redshift, and dbt. Prioritizes data quality, idempotency, and observability.",
    model: "claude-sonnet-4-20250514", temperature: 0.1, maxSteps: 15,
    systemPromptPreview: "You are a senior data engineer. Design efficient data pipelines, write optimized SQL...",
    reviews: [
      { author: "data_lead", rating: 5, comment: "Saved us weeks on our Snowflake migration.", date: "2026-05-16" },
    ],
    createdAt: "2026-04-05",
  },
  {
    id: "mkt-005", icon: "💬", name: "Empathy Support Agent", slug: "empathy-support-agent",
    category: "Support", publisher: "CareBot Inc", rating: 4.3, reviewCount: 156, downloads: 4200,
    shortDescription: "Customer-facing support with empathy and escalation awareness.",
    fullDescription: "Customer support agent trained for empathetic, accurate responses. Features knowledge base lookup, ticket categorization, sentiment analysis, and smart escalation. Maintains a warm, professional tone while resolving issues efficiently.",
    model: "gpt-4o", temperature: 0.5, maxSteps: 10,
    systemPromptPreview: "You are a friendly, knowledgeable customer support agent. Listen carefully, provide accurate answers...",
    reviews: [
      { author: "cs_manager", rating: 4, comment: "CSAT improved 15% since deployment. Escalation logic is solid.", date: "2026-05-11" },
      { author: "startup_ops", rating: 5, comment: "Handles 60% of our tickets autonomously now.", date: "2026-05-09" },
    ],
    createdAt: "2026-03-20",
  },
  {
    id: "mkt-006", icon: "🛡️", name: "Threat Hunter", slug: "threat-hunter",
    category: "Security", publisher: "Hidden Leaf Networks", rating: 4.6, reviewCount: 43, downloads: 980,
    shortDescription: "Vulnerability assessment, code audit, and compliance checking.",
    fullDescription: "Security auditor that analyzes code, configurations, and infrastructure for vulnerabilities. References OWASP Top 10, CWE, and NIST frameworks. Provides actionable remediation steps with severity ratings and compliance mapping.",
    model: "claude-sonnet-4-20250514", temperature: 0.1, maxSteps: 20,
    systemPromptPreview: "You are a security auditor. Analyze code, configurations, and infrastructure for vulnerabilities...",
    reviews: [
      { author: "sec_lead", rating: 5, comment: "Found 3 critical vulns our scanner missed. Impressive.", date: "2026-05-17" },
    ],
    createdAt: "2026-04-15",
  },
  {
    id: "mkt-007", icon: "⚡", name: "Incident Commander", slug: "incident-commander",
    category: "Operations", publisher: "OpsForge", rating: 4.4, reviewCount: 72, downloads: 1650,
    shortDescription: "Incident triage, coordination, and post-mortem generation.",
    fullDescription: "Operations coordinator that monitors system health, triages incidents, and coordinates responses across teams. Generates structured post-mortems, tracks SLOs, and provides runbook suggestions based on incident patterns.",
    model: "gpt-4", temperature: 0.3, maxSteps: 25,
    systemPromptPreview: "You are an incident commander. Monitor system health, triage incidents by severity...",
    reviews: [
      { author: "sre_team", rating: 4, comment: "MTTR dropped 30%. Post-mortems are comprehensive.", date: "2026-05-13" },
      { author: "devops_eng", rating: 5, comment: "Runbook suggestions alone are worth the install.", date: "2026-05-07" },
    ],
    createdAt: "2026-04-08",
  },
  {
    id: "mkt-008", icon: "🧬", name: "Academic Reviewer", slug: "academic-reviewer",
    category: "Research", publisher: "ScholarAI", rating: 4.2, reviewCount: 38, downloads: 720,
    shortDescription: "Peer-review style analysis of academic papers and proposals.",
    fullDescription: "Performs structured peer review of academic papers, grant proposals, and research manuscripts. Evaluates methodology, statistical rigor, novelty, and clarity. Provides detailed feedback with specific improvement suggestions.",
    model: "claude-opus-4-20250514", temperature: 0.4, maxSteps: 15,
    systemPromptPreview: "You are an academic peer reviewer. Evaluate research papers for methodology, statistical rigor...",
    reviews: [
      { author: "phd_student", rating: 4, comment: "Better feedback than most human reviewers I've had.", date: "2026-05-06" },
    ],
    createdAt: "2026-04-20",
  },
  {
    id: "mkt-009", icon: "🏗️", name: "Infra Provisioner", slug: "infra-provisioner",
    category: "Engineering", publisher: "CloudCraft", rating: 4.6, reviewCount: 51, downloads: 1340,
    shortDescription: "IaC generation for Terraform, Pulumi, and CloudFormation.",
    fullDescription: "Infrastructure-as-code specialist that generates production-ready Terraform, Pulumi, and CloudFormation templates. Follows security best practices, implements proper IAM policies, and includes monitoring and alerting configuration.",
    model: "claude-sonnet-4-20250514", temperature: 0.15, maxSteps: 20,
    systemPromptPreview: "You are an infrastructure engineer. Generate production-ready IaC with security best practices...",
    reviews: [
      { author: "platform_eng", rating: 5, comment: "The IAM policies it generates are actually least-privilege. Rare.", date: "2026-05-15" },
      { author: "devops_mgr", rating: 4, comment: "Great for bootstrapping. We still review before apply.", date: "2026-05-10" },
    ],
    createdAt: "2026-04-12",
  },
  {
    id: "mkt-010", icon: "📈", name: "Analytics Narrator", slug: "analytics-narrator",
    category: "Data", publisher: "InsightAI", rating: 4.1, reviewCount: 29, downloads: 650,
    shortDescription: "Transforms raw data into narrative insights and dashboards.",
    fullDescription: "Data storyteller that transforms raw metrics and datasets into compelling narratives. Generates executive summaries, identifies trends and anomalies, and suggests visualization approaches. Supports CSV, JSON, and SQL query results.",
    model: "gpt-4o", temperature: 0.6, maxSteps: 12,
    systemPromptPreview: "You are a data analyst and storyteller. Transform raw data into clear, actionable insights...",
    reviews: [
      { author: "bi_analyst", rating: 4, comment: "Saves time on weekly reports. Narrative quality is surprisingly good.", date: "2026-05-04" },
    ],
    createdAt: "2026-04-25",
  },
  {
    id: "mkt-011", icon: "📋", name: "Compliance Checker", slug: "compliance-checker",
    category: "Security", publisher: "RegTech Solutions", rating: 4.5, reviewCount: 35, downloads: 890,
    shortDescription: "SOC2, HIPAA, and GDPR compliance validation and gap analysis.",
    fullDescription: "Regulatory compliance specialist that audits systems and processes against SOC2, HIPAA, GDPR, and PCI-DSS frameworks. Generates gap analysis reports, remediation roadmaps, and evidence collection templates.",
    model: "claude-sonnet-4-20250514", temperature: 0.2, maxSteps: 18,
    systemPromptPreview: "You are a compliance auditor. Evaluate systems against SOC2, HIPAA, GDPR frameworks...",
    reviews: [
      { author: "compliance_officer", rating: 5, comment: "Cut our audit prep time in half.", date: "2026-05-16" },
      { author: "cto_startup", rating: 4, comment: "Great for SOC2 readiness. We passed our first audit.", date: "2026-05-02" },
    ],
    createdAt: "2026-04-18",
  },
  {
    id: "mkt-012", icon: "🎯", name: "Campaign Strategist", slug: "campaign-strategist",
    category: "Content", publisher: "GrowthHQ", rating: 4.4, reviewCount: 47, downloads: 1120,
    shortDescription: "Multi-channel marketing campaign planning and optimization.",
    fullDescription: "Marketing strategist that plans multi-channel campaigns across email, social, paid ads, and content marketing. Generates audience segments, A/B test hypotheses, budget allocations, and performance forecasts based on historical data.",
    model: "gpt-4o", temperature: 0.7, maxSteps: 15,
    systemPromptPreview: "You are a marketing strategist. Plan multi-channel campaigns with audience segmentation...",
    reviews: [
      { author: "growth_lead", rating: 5, comment: "The A/B test suggestions alone boosted our conversion 20%.", date: "2026-05-14" },
      { author: "mktg_coord", rating: 4, comment: "Helpful for ideation. We still validate budget forecasts manually.", date: "2026-05-05" },
    ],
    createdAt: "2026-04-22",
  },
  {
    id: "mkt-013", icon: "📡", name: "Omni-Channel Agent", slug: "omni-channel-agent",
    category: "Operations", publisher: "Hidden Leaf Networks", rating: 4.5, reviewCount: 58, downloads: 1480,
    shortDescription: "Deploy agent presence across Chat, Discord, Slack, and X simultaneously.",
    fullDescription: "Multi-platform communications agent that maintains consistent presence across Chat, Discord, Slack, and X. Synchronizes context between channels, identifies users across platforms, and manages threaded conversations with unified history.",
    model: "gpt-4o", temperature: 0.5, maxSteps: 15,
    systemPromptPreview: "You are a multi-platform communications agent. Maintain consistent presence across Chat, Discord, Slack, and X...",
    reviews: [
      { author: "community_mgr", rating: 5, comment: "Managing 4 platforms from one agent is a game changer.", date: "2026-05-18" },
      { author: "ops_lead", rating: 4, comment: "Context sync between Discord and Slack is seamless.", date: "2026-05-12" },
    ],
    createdAt: "2026-05-01",
  },
  {
    id: "mkt-014", icon: "📞", name: "Voice Concierge", slug: "voice-concierge",
    category: "Support", publisher: "VoiceFlow Labs", rating: 4.3, reviewCount: 41, downloads: 920,
    shortDescription: "Voice-driven phone agent for booking, intake, and outbound outreach.",
    fullDescription: "Professional phone agent handling inbound and outbound calls with natural conversation flow. Books appointments, conducts customer intake, and performs outreach campaigns. Integrates with calendar systems and CRMs for real-time availability checks.",
    model: "gpt-4o", temperature: 0.6, maxSteps: 10,
    systemPromptPreview: "You are a professional phone agent. Handle inbound and outbound calls with natural conversation flow...",
    reviews: [
      { author: "clinic_admin", rating: 4, comment: "Appointment no-shows dropped 40% since deploying.", date: "2026-05-16" },
      { author: "sales_dir", rating: 5, comment: "Outbound call quality rivals our best reps.", date: "2026-05-10" },
    ],
    createdAt: "2026-05-03",
  },
  {
    id: "mkt-015", icon: "🖥️", name: "Desktop Automator", slug: "desktop-automator",
    category: "Operations", publisher: "Hidden Leaf Networks", rating: 4.7, reviewCount: 33, downloads: 780,
    shortDescription: "Computer-use agent for screen interaction, app control, and RPA testing.",
    fullDescription: "Desktop automation agent powered by computer-use capabilities. Interacts with screen elements, controls applications, performs data entry, and executes test scenarios. Built-in approval gates and risk scoring for safe autonomous operation.",
    model: "claude-sonnet-4-20250514", temperature: 0.2, maxSteps: 30,
    systemPromptPreview: "You are a desktop automation agent. Interact with screen elements, control applications, perform data entry...",
    reviews: [
      { author: "qa_lead", rating: 5, comment: "Replaced 80% of our Selenium suite. Way more reliable.", date: "2026-05-17" },
      { author: "rpa_eng", rating: 4, comment: "Approval gates give us confidence to run unattended.", date: "2026-05-11" },
    ],
    createdAt: "2026-05-05",
  },
  {
    id: "mkt-016", icon: "🕷️", name: "DataSpider", slug: "dataspider",
    category: "Data", publisher: "CrawlTech", rating: 4.0, reviewCount: 67, downloads: 2100,
    shortDescription: "High-speed web scraping with structured extraction and rate limiting.",
    fullDescription: "Web data collection agent with structured extraction, pagination handling, and rate limit awareness. Supports proxy rotation, anti-bot evasion, and outputs clean validated data in JSON, CSV, or direct database insert formats.",
    model: "gpt-4o", temperature: 0.1, maxSteps: 20,
    systemPromptPreview: "You are a web scraping specialist. Collect data from websites efficiently with structured extraction...",
    reviews: [
      { author: "data_ops", rating: 4, comment: "Handles pagination better than any tool we've tried.", date: "2026-05-15" },
      { author: "mkt_analyst", rating: 4, comment: "Rate limiting logic prevented us from getting blocked.", date: "2026-05-09" },
    ],
    createdAt: "2026-05-02",
  },
  {
    id: "mkt-017", icon: "🔄", name: "Site Replicator", slug: "site-replicator",
    category: "Engineering", publisher: "Hidden Leaf Networks", rating: 4.8, reviewCount: 24, downloads: 560,
    shortDescription: "Reverse-engineer websites into clean React/Next.js components.",
    fullDescription: "Analyzes any website to extract design tokens, layout structures, and interaction patterns. Generates clean React/Next.js components with Tailwind CSS styling. Preserves visual fidelity while producing maintainable, accessible code.",
    model: "claude-sonnet-4-20250514", temperature: 0.3, maxSteps: 25,
    systemPromptPreview: "You are a site cloning specialist. Analyze websites to extract design tokens, layout structures...",
    reviews: [
      { author: "fe_lead", rating: 5, comment: "Turned a competitor's landing page into our own components in 10 minutes.", date: "2026-05-19" },
      { author: "agency_dev", rating: 5, comment: "Design token extraction alone saves hours per project.", date: "2026-05-14" },
    ],
    createdAt: "2026-05-08",
  },
  {
    id: "mkt-018", icon: "⚙️", name: "Flow Builder", slug: "flow-builder",
    category: "Engineering", publisher: "AutomateHQ", rating: 4.6, reviewCount: 52, downloads: 1750,
    shortDescription: "Generate n8n, Zapier, and Make workflows from natural language.",
    fullDescription: "Workflow automation specialist that converts natural language descriptions into executable workflow definitions for n8n, Zapier, and Make. Configures triggers, actions, conditional logic, and error handling. Outputs valid, importable JSON for each platform.",
    model: "claude-sonnet-4-20250514", temperature: 0.3, maxSteps: 15,
    systemPromptPreview: "You are a workflow automation specialist. Convert natural language descriptions into executable workflow definitions...",
    reviews: [
      { author: "nocode_lead", rating: 5, comment: "Generated a 12-step n8n workflow from one paragraph. Incredible.", date: "2026-05-18" },
      { author: "ops_mgr", rating: 4, comment: "Zapier export needed minor tweaks but saved days of work.", date: "2026-05-13" },
    ],
    createdAt: "2026-05-06",
  },
];

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function StarRating({ rating, size = "sm" }: { rating: number; size?: "sm" | "lg" }) {
  const px = size === "lg" ? "text-base" : "text-xs";
  return (
    <span className={`inline-flex gap-0.5 ${px}`} aria-label={`${rating.toFixed(1)} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} style={{ color: star <= Math.round(rating) ? "#FFB800" : "rgba(255,255,255,0.15)" }}>
          {star <= Math.round(rating) ? "\u2605" : "\u2606"}
        </span>
      ))}
    </span>
  );
}

function CategoryBadge({ category }: { category: Category }) {
  const color = CATEGORY_COLORS[category];
  return (
    <span
      className="inline-block rounded-full px-2 py-0.5 text-[10px] font-medium"
      style={{ background: `${color}20`, color, border: `1px solid ${color}30` }}
    >
      {category}
    </span>
  );
}

function formatDownloads(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/* ------------------------------------------------------------------ */
/*  Archetype Card                                                     */
/* ------------------------------------------------------------------ */

function ArchetypeCard({
  listing,
  isSelected,
  onSelect,
}: {
  listing: ArchetypeListing;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className="text-left w-full rounded-xl p-4 transition-all duration-200"
      style={{
        background: isSelected ? "rgba(0, 255, 170, 0.06)" : "rgba(15, 20, 35, 0.7)",
        backdropFilter: "blur(12px)",
        border: `1px solid ${isSelected ? "rgba(0, 255, 170, 0.25)" : "rgba(0, 255, 170, 0.08)"}`,
        boxShadow: isSelected ? "0 0 20px rgba(0, 255, 170, 0.1)" : "none",
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.borderColor = "rgba(0, 255, 170, 0.25)";
          e.currentTarget.style.boxShadow = "0 0 20px rgba(0, 255, 170, 0.1)";
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.borderColor = "rgba(0, 255, 170, 0.08)";
          e.currentTarget.style.boxShadow = "none";
        }
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="text-2xl">{listing.icon}</span>
        <CategoryBadge category={listing.category} />
      </div>
      <div className="text-sm font-semibold text-white mb-0.5">{listing.name}</div>
      <div className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.4)" }}>
        {listing.publisher}
      </div>
      <p className="text-xs mt-2 leading-relaxed" style={{ color: "rgba(255,255,255,0.5)" }}>
        {listing.shortDescription}
      </p>
      <div className="flex items-center gap-3 mt-3">
        <StarRating rating={listing.rating} />
        <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>
          {listing.rating.toFixed(1)} ({listing.reviewCount})
        </span>
        <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>
          {formatDownloads(listing.downloads)} installs
        </span>
      </div>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/*  Detail Panel                                                       */
/* ------------------------------------------------------------------ */

function DetailPanel({ listing, onClose }: { listing: ArchetypeListing; onClose: () => void }) {
  const [installing, setInstalling] = useState(false);

  const handleInstall = () => {
    setInstalling(true);
    setTimeout(() => setInstalling(false), 1500);
  };

  return (
    <HudPanel
      title={listing.name}
      subtitle={listing.slug}
      actions={
        <button className="text-xs text-white/30 hover:text-white/60 transition-colors" onClick={onClose}>
          Close
        </button>
      }
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div
            className="h-12 w-12 rounded-xl flex items-center justify-center text-2xl"
            style={{ background: "rgba(0, 255, 170, 0.08)", border: "1px solid rgba(0, 255, 170, 0.15)" }}
          >
            {listing.icon}
          </div>
          <div>
            <CategoryBadge category={listing.category} />
            <div className="flex items-center gap-2 mt-1">
              <StarRating rating={listing.rating} size="lg" />
              <span className="text-xs" style={{ color: "rgba(255,255,255,0.4)" }}>
                {listing.rating.toFixed(1)} ({listing.reviewCount} reviews)
              </span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="flex gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Downloads</div>
            <div className="text-sm font-mono text-white">{listing.downloads.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Publisher</div>
            <div className="text-sm text-white">{listing.publisher}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Added</div>
            <div className="text-sm font-mono text-white">{listing.createdAt}</div>
          </div>
        </div>

        {/* Description */}
        <div>
          <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "rgba(255,255,255,0.3)" }}>Description</div>
          <p className="text-xs leading-relaxed" style={{ color: "rgba(255,255,255,0.6)" }}>
            {listing.fullDescription}
          </p>
        </div>

        {/* Config */}
        <div className="grid grid-cols-3 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Model</div>
            <div className="text-xs font-mono text-white/70">{listing.model}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Temperature</div>
            <div className="text-xs font-mono text-white/70">{listing.temperature}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Max Steps</div>
            <div className="text-xs font-mono text-white/70">{listing.maxSteps}</div>
          </div>
        </div>

        {/* System Prompt Preview */}
        <div>
          <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "rgba(255,255,255,0.3)" }}>System Prompt Preview</div>
          <div
            className="text-xs font-mono leading-relaxed rounded-lg p-3"
            style={{ color: "rgba(255,255,255,0.4)", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
          >
            {listing.systemPromptPreview}
          </div>
        </div>

        {/* Install Button */}
        <button
          className="hud-btn hud-btn--primary w-full"
          onClick={handleInstall}
          disabled={installing}
        >
          {installing ? "Installing..." : "Install Archetype"}
        </button>

        {/* Reviews */}
        {listing.reviews.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: "rgba(255,255,255,0.3)" }}>
              Reviews ({listing.reviews.length})
            </div>
            <div className="space-y-2">
              {listing.reviews.map((review, i) => (
                <div
                  key={i}
                  className="rounded-lg p-3"
                  style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-white/60">{review.author}</span>
                      <StarRating rating={review.rating} />
                    </div>
                    <span className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.2)" }}>
                      {review.date}
                    </span>
                  </div>
                  <p className="text-xs" style={{ color: "rgba(255,255,255,0.5)" }}>{review.comment}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </HudPanel>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Page                                                          */
/* ------------------------------------------------------------------ */

export function MarketplacePage() {
  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState<"All" | Category>("All");
  const [sortBy, setSortBy] = useState<SortOption>("downloads");
  const [selected, setSelected] = useState<ArchetypeListing | null>(null);

  const filtered = useMemo(() => {
    let results = MOCK_LISTINGS;

    // Category filter
    if (activeCategory !== "All") {
      results = results.filter((l) => l.category === activeCategory);
    }

    // Search filter
    if (search.trim()) {
      const q = search.toLowerCase();
      results = results.filter(
        (l) => l.name.toLowerCase().includes(q) || l.shortDescription.toLowerCase().includes(q)
      );
    }

    // Sort
    switch (sortBy) {
      case "downloads":
        results = [...results].sort((a, b) => b.downloads - a.downloads);
        break;
      case "rating":
        results = [...results].sort((a, b) => b.rating - a.rating);
        break;
      case "newest":
        results = [...results].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
        break;
    }

    return results;
  }, [search, activeCategory, sortBy]);

  return (
    <div>
      <HudTopBar
        title="Marketplace"
        subtitle="Browse, install, and rate community archetypes"
      />

      <div className="p-6">
        {/* Hero / Search Section */}
        <div
          className="rounded-xl p-6 mb-6"
          style={{
            background: "linear-gradient(135deg, rgba(0, 255, 170, 0.04) 0%, rgba(168, 85, 247, 0.04) 100%)",
            border: "1px solid rgba(0, 255, 170, 0.08)",
          }}
        >
          <h2 className="text-lg font-semibold text-white mb-1">Archetype Marketplace</h2>
          <p className="text-xs mb-4" style={{ color: "rgba(255,255,255,0.4)" }}>
            Discover pre-built agent archetypes from the community. Install with one click.
          </p>

          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search */}
            <div className="flex-1">
              <input
                className="hud-input w-full"
                placeholder="Search archetypes by name or description..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            {/* Sort */}
            <select
              className="hud-input px-3"
              style={{ minWidth: 160, cursor: "pointer" }}
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortOption)}
            >
              <option value="downloads">Most Downloads</option>
              <option value="rating">Highest Rated</option>
              <option value="newest">Newest</option>
            </select>
          </div>

          {/* Category Filters */}
          <div className="flex flex-wrap gap-2 mt-4">
            {CATEGORIES.map((cat) => {
              const isActive = activeCategory === cat;
              const color = cat === "All" ? "#00FFAA" : CATEGORY_COLORS[cat];
              return (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className="rounded-full px-3 py-1 text-xs font-medium transition-all duration-200"
                  style={{
                    background: isActive ? `${color}20` : "rgba(255,255,255,0.03)",
                    color: isActive ? color : "rgba(255,255,255,0.4)",
                    border: `1px solid ${isActive ? `${color}40` : "rgba(255,255,255,0.06)"}`,
                  }}
                >
                  {cat}
                </button>
              );
            })}
          </div>
        </div>

        {/* Content Grid */}
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
          {/* Card Grid */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono" style={{ color: "rgba(255,255,255,0.3)" }}>
                {filtered.length} archetype{filtered.length !== 1 ? "s" : ""} found
              </span>
            </div>

            {filtered.length > 0 ? (
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {filtered.map((listing) => (
                  <ArchetypeCard
                    key={listing.id}
                    listing={listing}
                    isSelected={selected?.id === listing.id}
                    onSelect={() => setSelected(listing)}
                  />
                ))}
              </div>
            ) : (
              <div
                className="rounded-xl py-16 text-center"
                style={{ background: "rgba(15, 20, 35, 0.7)", border: "1px solid rgba(0, 255, 170, 0.08)" }}
              >
                <p className="text-sm text-white/30">No archetypes match your search</p>
                <p className="text-xs text-white/15 mt-1">Try a different search term or category</p>
              </div>
            )}
          </div>

          {/* Detail Panel */}
          {selected ? (
            <DetailPanel listing={selected} onClose={() => setSelected(null)} />
          ) : (
            <HudPanel className="flex items-center justify-center min-h-[300px]">
              <div className="text-center">
                <p className="text-sm text-white/30">Select an archetype</p>
                <p className="text-xs text-white/15 mt-1">Click a card to view details and install</p>
              </div>
            </HudPanel>
          )}
        </div>
      </div>
    </div>
  );
}

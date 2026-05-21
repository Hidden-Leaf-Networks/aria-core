import { useState } from "react";
import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { agents, type Agent } from "@/lib/api";

const MODELS = ["gpt-4", "gpt-4o", "claude-sonnet-4-20250514", "claude-opus-4-20250514", "grok-2-latest"];

export function AgentsPage() {
  const { data: agentList, refetch } = useApi(() => agents.list(), [], 10000);
  const [showCreate, setShowCreate] = useState(false);
  const [selected, setSelected] = useState<Agent | null>(null);

  return (
    <div>
      <HudTopBar
        title="Agents"
        subtitle="Agent registry — manage your agent workforce"
        actions={
          <button className="hud-btn hud-btn--primary" onClick={() => { setShowCreate(true); setSelected(null); }}>
            + Register Agent
          </button>
        }
      />
      <div className="p-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_400px]">
          {/* Agent grid */}
          <div>
            {agentList && agentList.length > 0 ? (
              <div className="grid gap-3 sm:grid-cols-2">
                {agentList.map((agent) => (
                  <button
                    key={agent.id}
                    onClick={() => { setSelected(agent); setShowCreate(false); }}
                    className={`hud-panel text-left transition-all ${
                      selected?.id === agent.id ? "hud-panel--selected" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-hud-accent2 to-hud-accent3 flex items-center justify-center text-xs font-bold text-white">
                        {agent.name.charAt(0).toUpperCase()}
                      </div>
                      <HudStatusPill state={agent.status} />
                    </div>
                    <div className="text-sm font-medium text-white">{agent.name}</div>
                    <div className="text-[10px] font-mono text-white/30">{agent.slug}</div>
                    <div className="mt-2 flex gap-3 text-[10px] text-white/20">
                      <span>{agent.model}</span>
                      <span>{agent.executions} runs</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <HudPanel className="py-12 text-center">
                <p className="text-sm text-white/30">No agents registered</p>
                <p className="text-xs text-white/15 mt-1">Click "Register Agent" to add your first agent</p>
              </HudPanel>
            )}
          </div>

          {/* Right panel */}
          {showCreate ? (
            <CreateAgentForm
              onCreated={() => { setShowCreate(false); refetch(); }}
              onCancel={() => setShowCreate(false)}
            />
          ) : selected ? (
            <AgentDetail agent={selected} onUpdate={refetch} />
          ) : (
            <HudPanel className="flex items-center justify-center min-h-[300px]">
              <p className="text-sm text-white/30">Select or register an agent</p>
            </HudPanel>
          )}
        </div>
      </div>
    </div>
  );
}

const AGENT_TEMPLATES = [
  {
    label: "Research Analyst", icon: "🔍",
    name: "Research Analyst", slug: "research-analyst",
    description: "Deep research and analysis across multiple sources. Synthesizes findings into structured reports.",
    model: "claude-sonnet-4-20250514", maxSteps: 20, temperature: 0.3,
    systemPrompt: "You are a meticulous research analyst. Gather information from multiple sources, cross-reference findings, identify patterns, and produce clear, well-structured reports with citations.",
  },
  {
    label: "Code Assistant", icon: "💻",
    name: "Code Assistant", slug: "code-assistant",
    description: "Software development assistant for code generation, review, and debugging.",
    model: "claude-sonnet-4-20250514", maxSteps: 15, temperature: 0.2,
    systemPrompt: "You are an expert software engineer. Write clean, well-tested code. Follow best practices for the target language. Explain your reasoning and suggest improvements.",
  },
  {
    label: "Content Writer", icon: "✍️",
    name: "Content Writer", slug: "content-writer",
    description: "Professional content creation with brand voice consistency and SEO awareness.",
    model: "gpt-4o", maxSteps: 10, temperature: 0.8,
    systemPrompt: "You are a professional content writer. Create engaging, well-structured content that matches the brand voice. Optimize for readability and SEO. Adapt tone to the target audience.",
  },
  {
    label: "Data Engineer", icon: "📊",
    name: "Data Engineer", slug: "data-engineer",
    description: "Data pipeline design, SQL generation, schema design, and ETL orchestration.",
    model: "claude-sonnet-4-20250514", maxSteps: 15, temperature: 0.1,
    systemPrompt: "You are a senior data engineer. Design efficient data pipelines, write optimized SQL, and build robust ETL processes. Prioritize data quality, idempotency, and observability.",
  },
  {
    label: "Customer Support", icon: "💬",
    name: "Customer Support Agent", slug: "support-agent",
    description: "Customer-facing support with empathy, knowledge base lookup, and escalation awareness.",
    model: "gpt-4o", maxSteps: 10, temperature: 0.5,
    systemPrompt: "You are a friendly, knowledgeable customer support agent. Listen carefully, provide accurate answers, and escalate when needed. Always maintain a helpful and empathetic tone.",
  },
  {
    label: "Security Auditor", icon: "🛡️",
    name: "Security Auditor", slug: "security-auditor",
    description: "Security analysis, vulnerability assessment, and compliance checking.",
    model: "claude-sonnet-4-20250514", maxSteps: 20, temperature: 0.1,
    systemPrompt: "You are a security auditor. Analyze code, configurations, and infrastructure for vulnerabilities. Reference OWASP, CWE, and NIST standards. Provide actionable remediation steps with severity ratings.",
  },
  {
    label: "Ops Coordinator", icon: "⚡",
    name: "Ops Coordinator", slug: "ops-coordinator",
    description: "Operations orchestration — monitors, triages, and coordinates responses across systems.",
    model: "gpt-4", maxSteps: 25, temperature: 0.3,
    systemPrompt: "You are an operations coordinator. Monitor system health, triage incidents, and coordinate responses. Prioritize by severity. Communicate clearly and escalate when thresholds are breached.",
  },
];

function CreateAgentForm({ onCreated, onCancel }: { onCreated: () => void; onCancel: () => void }) {
  const [showTemplates, setShowTemplates] = useState(true);
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [description, setDescription] = useState("");
  const [model, setModel] = useState("gpt-4");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [maxSteps, setMaxSteps] = useState(10);
  const [temperature, setTemperature] = useState(0.7);
  const [saving, setSaving] = useState(false);

  const applyTemplate = (t: typeof AGENT_TEMPLATES[0]) => {
    setName(t.name);
    setSlug(t.slug);
    setDescription(t.description);
    setModel(t.model);
    setSystemPrompt(t.systemPrompt);
    setMaxSteps(t.maxSteps);
    setTemperature(t.temperature);
    setShowTemplates(false);
  };

  if (showTemplates) {
    return (
      <HudPanel title="Register Agent" subtitle="Choose a template or start blank" selected>
        <div className="space-y-2 mb-3 max-h-[400px] overflow-y-auto hud-scroll">
          {AGENT_TEMPLATES.map((t, i) => (
            <button
              key={i}
              onClick={() => applyTemplate(t)}
              className="w-full text-left rounded-lg border border-white/[0.06] hover:border-hud-accent2/30 bg-white/[0.02] hover:bg-white/[0.04] p-3 transition-all"
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{t.icon}</span>
                <div>
                  <div className="text-sm font-medium text-white">{t.label}</div>
                  <div className="text-[10px] text-white/30">{t.model} — temp {t.temperature} — {t.maxSteps} steps</div>
                </div>
              </div>
            </button>
          ))}
        </div>
        <div className="flex gap-2 pt-2 border-t border-white/5">
          <button className="hud-btn hud-btn--secondary flex-1 text-xs" onClick={() => setShowTemplates(false)}>Start Blank</button>
          <button className="hud-btn hud-btn--ghost text-xs" onClick={onCancel}>Cancel</button>
        </div>
      </HudPanel>
    );
  }

  const submit = async () => {
    if (!name) return;
    setSaving(true);
    try {
      await agents.register({
        name,
        slug: slug || name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
        description,
        model,
        system_prompt: systemPrompt || undefined,
        max_steps: maxSteps,
        temperature,
      });
      onCreated();
    } finally {
      setSaving(false);
    }
  };

  return (
    <HudPanel title="Register Agent" selected>
      <div className="space-y-3">
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Name</label>
          <input className="hud-input mt-1" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Kairo" autoFocus />
        </div>
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Slug</label>
          <input className="hud-input mt-1 font-mono" value={slug} onChange={(e) => setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ""))} placeholder="auto-generated from name" />
        </div>
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Description</label>
          <textarea className="hud-input mt-1 h-16 resize-none" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What does this agent do?" />
        </div>
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Model</label>
          <select className="hud-select mt-1" value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-white/40 uppercase tracking-wider">Max Steps</label>
            <input className="hud-input mt-1" type="number" value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} min={1} max={50} />
          </div>
          <div>
            <label className="text-xs text-white/40 uppercase tracking-wider">Temperature</label>
            <input className="hud-input mt-1" type="number" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} min={0} max={2} step={0.1} />
          </div>
        </div>
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">System Prompt</label>
          <textarea className="hud-input mt-1 h-20 resize-none font-mono text-xs" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} placeholder="Optional system prompt..." />
        </div>
        <div className="flex gap-2 pt-2">
          <button className="hud-btn hud-btn--primary flex-1" onClick={submit} disabled={saving || !name}>
            {saving ? "Registering..." : "Register"}
          </button>
          <button className="hud-btn hud-btn--secondary" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </HudPanel>
  );
}

function AgentDetail({ agent, onUpdate }: { agent: Agent; onUpdate: () => void }) {
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    setDeleting(true);
    try { await agents.delete(agent.id); onUpdate(); } finally { setDeleting(false); }
  };

  return (
    <HudPanel
      title={agent.name}
      subtitle={agent.slug}
      actions={
        <button className="hud-btn hud-btn--danger text-xs" onClick={handleDelete} disabled={deleting}>
          {deleting ? "..." : "Delete"}
        </button>
      }
    >
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-hud-accent2 to-hud-accent3 flex items-center justify-center text-lg font-bold text-white">
            {agent.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <HudStatusPill state={agent.status} />
            <div className="text-xs text-white/30 mt-1">{agent.executions} executions</div>
          </div>
        </div>

        {agent.description && <p className="text-xs text-white/50">{agent.description}</p>}

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-white/30 uppercase">Model</label>
            <div className="text-xs font-mono text-white/70">{agent.model}</div>
          </div>
          <div>
            <label className="text-[10px] text-white/30 uppercase">Temperature</label>
            <div className="text-xs font-mono text-white/70">{agent.temperature}</div>
          </div>
          <div>
            <label className="text-[10px] text-white/30 uppercase">Max Steps</label>
            <div className="text-xs font-mono text-white/70">{agent.max_steps}</div>
          </div>
          <div>
            <label className="text-[10px] text-white/30 uppercase">Created By</label>
            <div className="text-xs font-mono text-white/70">{agent.created_by}</div>
          </div>
        </div>

        {agent.system_prompt && (
          <div>
            <label className="text-[10px] text-white/30 uppercase">System Prompt</label>
            <div className="text-xs font-mono text-white/40 mt-1 bg-white/[0.02] rounded p-2 max-h-[150px] overflow-y-auto hud-scroll">
              {agent.system_prompt}
            </div>
          </div>
        )}

        {agent.allowed_skills.length > 0 && (
          <div>
            <label className="text-[10px] text-white/30 uppercase">Allowed Skills</label>
            <div className="flex flex-wrap gap-1 mt-1">
              {agent.allowed_skills.map((s) => (
                <span key={s} className="px-2 py-0.5 rounded bg-white/5 text-[10px] font-mono text-white/40">{s}</span>
              ))}
            </div>
          </div>
        )}

        <div className="pt-2 border-t border-white/5 text-[10px] text-white/15 font-mono">
          ID: {agent.id} — {new Date(agent.created_at).toLocaleString()}
        </div>
      </div>
    </HudPanel>
  );
}

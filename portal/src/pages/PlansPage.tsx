import { useState } from "react";
import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { plans, type Plan } from "@/lib/api";

export function PlansPage() {
  const { data: planList, refetch } = useApi(() => plans.list({ limit: 100 }), [], 10000);
  const [showWizard, setShowWizard] = useState(false);
  const [selected, setSelected] = useState<Plan | null>(null);

  const byState = (planList ?? []).reduce<Record<string, number>>((acc, p) => {
    acc[p.state] = (acc[p.state] || 0) + 1;
    return acc;
  }, {});

  return (
    <div>
      <HudTopBar
        title="Plans"
        subtitle="Plan lifecycle management"
        actions={
          <button className="hud-btn hud-btn--primary" onClick={() => { setShowWizard(true); setSelected(null); }}>
            + New Plan
          </button>
        }
      />
      <div className="p-6 space-y-6">
        {/* State breakdown */}
        {Object.keys(byState).length > 0 && (
          <div className="flex gap-3 flex-wrap">
            {Object.entries(byState).map(([state, count]) => (
              <div key={state} className="hud-panel flex items-center gap-2 px-3 py-2">
                <HudStatusPill state={state} />
                <span className="text-sm font-mono text-white/60">{count}</span>
              </div>
            ))}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[1fr_420px]">
          {/* Plan table */}
          <HudPanel title="All Plans" subtitle={`${planList?.length ?? 0} plans`}>
            {planList && planList.length > 0 ? (
              <div className="space-y-2">
                {planList.map((plan) => (
                  <button
                    key={plan.id}
                    onClick={() => { setSelected(plan); setShowWizard(false); }}
                    className={`w-full text-left rounded-lg border p-3 transition-all ${
                      selected?.id === plan.id
                        ? "border-hud-accent/30 bg-white/[0.04]"
                        : "border-transparent hover:bg-white/[0.02]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-medium text-white text-sm">{plan.name}</div>
                      <HudStatusPill state={plan.state} />
                    </div>
                    <div className="mt-1 flex gap-4 text-xs text-white/30">
                      <span>{plan.actions.length} actions</span>
                      <span>Risk: {plan.aggregate_risk_score ?? "—"}</span>
                      <span>By: {plan.created_by}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/30 py-8 text-center">No plans yet</p>
            )}
          </HudPanel>

          {/* Right panel: wizard or detail */}
          {showWizard ? (
            <PlanWizard
              onCreated={() => { setShowWizard(false); refetch(); }}
              onCancel={() => setShowWizard(false)}
            />
          ) : selected ? (
            <PlanDetail plan={selected} onUpdate={refetch} />
          ) : (
            <HudPanel className="flex items-center justify-center min-h-[300px]">
              <p className="text-sm text-white/30">Select a plan or create a new one</p>
            </HudPanel>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Plan Creation Wizard                                                */
/* ------------------------------------------------------------------ */

interface WizardAction {
  name: string;
  skill_name: string;
  description: string;
  dependencies: number[];
}

const PLAN_TEMPLATES: { label: string; icon: string; name: string; description: string; actions: WizardAction[] }[] = [
  {
    label: "CI/CD Pipeline", icon: "🚀",
    name: "CI/CD Deploy Pipeline",
    description: "Build, test, and deploy to production with approval gate",
    actions: [
      { name: "Build Docker Image", skill_name: "docker_build", description: "Build container from Dockerfile", dependencies: [] },
      { name: "Run Test Suite", skill_name: "pytest", description: "Run all unit and integration tests", dependencies: [0] },
      { name: "Security Scan", skill_name: "trivy_scan", description: "Scan image for vulnerabilities", dependencies: [0] },
      { name: "Push to Registry", skill_name: "docker_push", description: "Push to container registry", dependencies: [1, 2] },
      { name: "Deploy to Staging", skill_name: "helm_upgrade", description: "Deploy to staging cluster", dependencies: [3] },
      { name: "Deploy to Production", skill_name: "helm_upgrade_prod", description: "Deploy to production cluster", dependencies: [4] },
    ],
  },
  {
    label: "Research Agent", icon: "🔍",
    name: "Research & Analysis Pipeline",
    description: "Multi-source research with synthesis and report generation",
    actions: [
      { name: "Web Research", skill_name: "web_search", description: "Search and scrape relevant sources", dependencies: [] },
      { name: "Document Analysis", skill_name: "doc_parse", description: "Extract key findings from documents", dependencies: [] },
      { name: "Data Synthesis", skill_name: "synthesize", description: "Combine findings into coherent analysis", dependencies: [0, 1] },
      { name: "Generate Report", skill_name: "generate_report", description: "Create formatted research report", dependencies: [2] },
    ],
  },
  {
    label: "Content Pipeline", icon: "✍️",
    name: "Content Creation Pipeline",
    description: "Draft, review, optimize, and publish content",
    actions: [
      { name: "Generate Draft", skill_name: "llm_generate", description: "Create initial content draft", dependencies: [] },
      { name: "SEO Optimization", skill_name: "seo_optimize", description: "Optimize for search engines", dependencies: [0] },
      { name: "Tone Review", skill_name: "tone_check", description: "Verify brand voice and tone", dependencies: [0] },
      { name: "Final Edit", skill_name: "edit_polish", description: "Final grammar and style pass", dependencies: [1, 2] },
      { name: "Publish", skill_name: "publish", description: "Push to CMS/platform", dependencies: [3] },
    ],
  },
  {
    label: "Data ETL", icon: "📊",
    name: "Data ETL Pipeline",
    description: "Extract, transform, validate, and load data",
    actions: [
      { name: "Extract Source Data", skill_name: "extract", description: "Pull data from source systems", dependencies: [] },
      { name: "Transform & Clean", skill_name: "transform", description: "Normalize, dedupe, and clean data", dependencies: [0] },
      { name: "Validate Schema", skill_name: "validate", description: "Validate against target schema", dependencies: [1] },
      { name: "Load to Warehouse", skill_name: "load", description: "Insert into data warehouse", dependencies: [2] },
    ],
  },
  {
    label: "Incident Response", icon: "🚨",
    name: "Incident Response Runbook",
    description: "Automated incident triage, diagnosis, mitigation, and postmortem",
    actions: [
      { name: "Collect Metrics", skill_name: "metrics_collect", description: "Pull logs, metrics, and traces", dependencies: [] },
      { name: "Root Cause Analysis", skill_name: "rca_analyze", description: "Identify probable root cause", dependencies: [0] },
      { name: "Apply Mitigation", skill_name: "mitigate", description: "Execute mitigation steps", dependencies: [1] },
      { name: "Verify Recovery", skill_name: "verify_health", description: "Confirm services are healthy", dependencies: [2] },
      { name: "Generate Postmortem", skill_name: "postmortem", description: "Create incident report", dependencies: [3] },
    ],
  },
  {
    label: "Customer Onboarding", icon: "👤",
    name: "Customer Onboarding Flow",
    description: "Provision tenant, configure, seed data, and notify",
    actions: [
      { name: "Create Tenant", skill_name: "create_tenant", description: "Provision new tenant in the system", dependencies: [] },
      { name: "Configure Defaults", skill_name: "set_config", description: "Apply default model and policy settings", dependencies: [0] },
      { name: "Seed Demo Data", skill_name: "seed_data", description: "Load sample plans and agents", dependencies: [1] },
      { name: "Send Welcome Email", skill_name: "send_email", description: "Notify customer with credentials", dependencies: [2] },
    ],
  },
];

function PlanWizard({ onCreated, onCancel }: { onCreated: () => void; onCancel: () => void }) {
  const [step, setStep] = useState(-1); // -1 = template picker
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [actions, setActions] = useState<WizardAction[]>([]);
  const [saving, setSaving] = useState(false);

  const applyTemplate = (t: typeof PLAN_TEMPLATES[0]) => {
    setName(t.name);
    setDescription(t.description);
    setActions(t.actions.map((a) => ({ ...a })));
    setStep(0);
  };

  const startBlank = () => {
    setName("");
    setDescription("");
    setActions([]);
    setStep(0);
  };

  const addAction = () => {
    setActions([...actions, { name: "", skill_name: "", description: "", dependencies: [] }]);
  };

  const updateAction = (idx: number, field: keyof WizardAction, value: string | number[]) => {
    const updated = [...actions];
    (updated[idx] as unknown as Record<string, unknown>)[field] = value;
    setActions(updated);
  };

  const removeAction = (idx: number) => {
    setActions(actions.filter((_, i) => i !== idx).map((a) => ({
      ...a,
      dependencies: a.dependencies.filter((d) => d !== idx).map((d) => (d > idx ? d - 1 : d)),
    })));
  };

  const toggleDep = (actionIdx: number, depIdx: number) => {
    const deps = actions[actionIdx].dependencies;
    const updated = deps.includes(depIdx) ? deps.filter((d) => d !== depIdx) : [...deps, depIdx];
    updateAction(actionIdx, "dependencies", updated);
  };

  const submit = async () => {
    setSaving(true);
    try {
      await plans.create({
        name,
        description,
        actions: actions.map((a) => ({
          name: a.name,
          skill_name: a.skill_name,
          description: a.description,
          dependencies: a.dependencies,
        })),
      });
      onCreated();
    } finally {
      setSaving(false);
    }
  };

  const steps = ["Details", "Actions", "Dependencies", "Review"];

  // Template picker (step -1)
  if (step === -1) {
    return (
      <HudPanel title="Create Plan" subtitle="Choose a template or start blank" selected>
        <div className="space-y-2 mb-3">
          {PLAN_TEMPLATES.map((t, i) => (
            <button
              key={i}
              onClick={() => applyTemplate(t)}
              className="w-full text-left rounded-lg border border-white/[0.06] hover:border-hud-accent/30 bg-white/[0.02] hover:bg-white/[0.04] p-3 transition-all"
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{t.icon}</span>
                <div>
                  <div className="text-sm font-medium text-white">{t.label}</div>
                  <div className="text-[10px] text-white/30">{t.actions.length} actions — {t.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
        <div className="flex gap-2 pt-2 border-t border-white/5">
          <button className="hud-btn hud-btn--secondary flex-1 text-xs" onClick={startBlank}>Start Blank</button>
          <button className="hud-btn hud-btn--ghost text-xs" onClick={onCancel}>Cancel</button>
        </div>
      </HudPanel>
    );
  }

  return (
    <HudPanel title="Create Plan" subtitle={`Step ${step + 1}: ${steps[step]}`} selected>
      {/* Step indicator */}
      <div className="flex gap-1 mb-4">
        {steps.map((s, i) => (
          <div
            key={s}
            className={`h-1 flex-1 rounded-full transition-all ${
              i <= step ? "bg-hud-accent" : "bg-white/10"
            }`}
          />
        ))}
      </div>

      {/* Step 0: Name & Description */}
      {step === 0 && (
        <div className="space-y-4">
          <div>
            <label className="text-xs text-white/40 uppercase tracking-wider">Plan Name</label>
            <input className="hud-input mt-1" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Deploy to Production" autoFocus />
          </div>
          <div>
            <label className="text-xs text-white/40 uppercase tracking-wider">Description</label>
            <textarea className="hud-input mt-1 h-20 resize-none" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What does this plan accomplish?" />
          </div>
        </div>
      )}

      {/* Step 1: Actions */}
      {step === 1 && (
        <div className="space-y-3">
          {actions.map((action, idx) => (
            <div key={idx} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-white/30 font-mono">Action {idx}</span>
                <button className="text-xs text-hud-error/60 hover:text-hud-error" onClick={() => removeAction(idx)}>Remove</button>
              </div>
              <input className="hud-input text-xs" value={action.name} onChange={(e) => updateAction(idx, "name", e.target.value)} placeholder="Action name" />
              <input className="hud-input text-xs font-mono" value={action.skill_name} onChange={(e) => updateAction(idx, "skill_name", e.target.value)} placeholder="Skill name (e.g. docker_build)" />
            </div>
          ))}
          <button className="hud-btn hud-btn--secondary w-full text-xs" onClick={addAction}>+ Add Action</button>
        </div>
      )}

      {/* Step 2: Dependencies */}
      {step === 2 && (
        <div className="space-y-3">
          {actions.length < 2 ? (
            <p className="text-sm text-white/30 py-4 text-center">Need 2+ actions for dependencies</p>
          ) : (
            actions.map((action, idx) => (
              <div key={idx} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                <div className="text-xs font-medium text-white mb-2">{action.name || `Action ${idx}`}</div>
                <div className="text-[10px] text-white/30 mb-1">Depends on:</div>
                <div className="flex flex-wrap gap-1">
                  {actions.map((dep, depIdx) => {
                    if (depIdx === idx) return null;
                    const isActive = action.dependencies.includes(depIdx);
                    return (
                      <button
                        key={depIdx}
                        onClick={() => toggleDep(idx, depIdx)}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
                          isActive
                            ? "bg-hud-accent/20 text-hud-accent border border-hud-accent/30"
                            : "bg-white/5 text-white/30 border border-transparent hover:bg-white/10"
                        }`}
                      >
                        {dep.name || `Action ${depIdx}`}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Step 3: Review */}
      {step === 3 && (
        <div className="space-y-3">
          <div className="text-sm text-white font-medium">{name}</div>
          {description && <div className="text-xs text-white/40">{description}</div>}
          <div className="space-y-1">
            {actions.map((a, idx) => (
              <div key={idx} className="flex items-center gap-2 text-xs rounded bg-white/[0.03] p-2">
                <span className="text-white/20 font-mono w-5">{idx}</span>
                <span className="text-white/80">{a.name}</span>
                <span className="text-white/30 font-mono">{a.skill_name}</span>
                {a.dependencies.length > 0 && (
                  <span className="text-hud-accent3/60 text-[10px]">← {a.dependencies.join(", ")}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex gap-2 mt-4 pt-3 border-t border-white/5">
        {step > 0 && (
          <button className="hud-btn hud-btn--ghost text-xs" onClick={() => setStep(step - 1)}>Back</button>
        )}
        <div className="flex-1" />
        {step < 3 ? (
          <button
            className="hud-btn hud-btn--primary text-xs"
            onClick={() => setStep(step + 1)}
            disabled={step === 0 && !name}
          >
            Next
          </button>
        ) : (
          <button className="hud-btn hud-btn--primary text-xs" onClick={submit} disabled={saving || !name || actions.length === 0}>
            {saving ? "Creating..." : "Create Plan"}
          </button>
        )}
        <button className="hud-btn hud-btn--ghost text-xs" onClick={onCancel}>Cancel</button>
      </div>
    </HudPanel>
  );
}

/* ------------------------------------------------------------------ */
/* Plan Detail                                                         */
/* ------------------------------------------------------------------ */

function PlanDetail({ plan, onUpdate }: { plan: Plan; onUpdate: () => void }) {
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await plans.delete(plan.id);
      onUpdate();
    } finally {
      setDeleting(false);
    }
  };

  return (
    <HudPanel
      title={plan.name}
      subtitle={`${plan.actions.length} actions — ${plan.state}`}
      actions={
        <button className="hud-btn hud-btn--danger text-xs" onClick={handleDelete} disabled={deleting}>
          {deleting ? "..." : "Delete"}
        </button>
      }
    >
      <div className="space-y-4">
        {plan.description && (
          <p className="text-xs text-white/50">{plan.description}</p>
        )}

        {/* Action timeline */}
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider mb-2 block">Actions</label>
          <div className="space-y-2">
            {plan.actions.map((action, idx) => (
              <div key={action.id} className="flex items-start gap-3 rounded-lg bg-white/[0.02] p-3">
                <div className="flex flex-col items-center">
                  <div className={`h-6 w-6 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    action.state === "completed" ? "bg-hud-success/20 text-hud-success" :
                    action.state === "failed" ? "bg-hud-error/20 text-hud-error" :
                    action.state === "executing" ? "bg-hud-warning/20 text-hud-warning" :
                    "bg-white/10 text-white/40"
                  }`}>
                    {idx}
                  </div>
                  {idx < plan.actions.length - 1 && <div className="w-px h-4 bg-white/10 mt-1" />}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">{action.name}</span>
                    <HudStatusPill state={action.state} />
                  </div>
                  {action.skill_name && (
                    <span className="text-[10px] font-mono text-white/30">{action.skill_name}</span>
                  )}
                  {action.error && (
                    <div className="text-xs text-hud-error mt-1">{action.error}</div>
                  )}
                  {action.execution_time_ms != null && (
                    <span className="text-[10px] text-white/20">{action.execution_time_ms}ms</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-2 border-t border-white/5 text-xs text-white/20 font-mono space-y-1">
          <div>ID: {plan.id}</div>
          <div>Created: {new Date(plan.created_at).toLocaleString()}</div>
          <div>By: {plan.created_by}</div>
        </div>
      </div>
    </HudPanel>
  );
}

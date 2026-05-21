import { useState, useCallback } from "react";
import { providers, agents } from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const PROVIDER_OPTIONS = [
  { key: "openai", label: "OpenAI", icon: "\uD83D\uDFE2", placeholder: "sk-..." },
  { key: "anthropic", label: "Anthropic", icon: "\uD83D\uDFE3", placeholder: "sk-ant-..." },
  { key: "xai", label: "xAI", icon: "\u26A1", placeholder: "xai-..." },
] as const;

const AGENT_ARCHETYPES = [
  { label: "Research Analyst", icon: "\uD83D\uDD0D", slug: "research-analyst", model: "claude-sonnet-4-20250514", description: "Deep research and analysis across multiple sources.", systemPrompt: "You are a meticulous research analyst. Gather information, cross-reference findings, and produce structured reports.", maxSteps: 20, temperature: 0.3 },
  { label: "Code Assistant", icon: "\uD83D\uDCBB", slug: "code-assistant", model: "claude-sonnet-4-20250514", description: "Software development assistant for code gen, review, and debugging.", systemPrompt: "You are an expert software engineer. Write clean, well-tested code and explain your reasoning.", maxSteps: 15, temperature: 0.2 },
  { label: "Content Writer", icon: "\u270D\uFE0F", slug: "content-writer", model: "gpt-4o", description: "Professional content creation with brand voice consistency.", systemPrompt: "You are a professional content writer. Create engaging, well-structured content optimized for readability.", maxSteps: 10, temperature: 0.8 },
  { label: "Customer Support", icon: "\uD83D\uDCAC", slug: "support-agent", model: "gpt-4o", description: "Customer-facing support with empathy and knowledge base lookup.", systemPrompt: "You are a friendly, knowledgeable customer support agent. Listen carefully and provide accurate answers.", maxSteps: 10, temperature: 0.5 },
];

const STEP_LABELS = ["Welcome", "Configure Provider", "Create Agent", "Run Agent"];

/* ------------------------------------------------------------------ */
/*  Main Page                                                          */
/* ------------------------------------------------------------------ */

export function OnboardingPage() {
  const [step, setStep] = useState(0);

  // Step 2 state — provider config
  const [selectedProvider, setSelectedProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [providerSaving, setProviderSaving] = useState(false);
  const [providerConfigured, setProviderConfigured] = useState(false);

  // Step 3 state — agent create
  const [selectedArchetype, setSelectedArchetype] = useState<number | null>(null);
  const [agentCreating, setAgentCreating] = useState(false);
  const [agentCreated, setAgentCreated] = useState(false);
  const [agentSlug, setAgentSlug] = useState("");

  // Step 4 state — test run
  const [userMessage, setUserMessage] = useState("");
  const [response, setResponse] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  /* --- Step handlers --- */

  const handleConfigureProvider = useCallback(async () => {
    if (!apiKey.trim()) return;
    setProviderSaving(true);
    try {
      await providers.configure({ provider: selectedProvider, api_key: apiKey.trim() });
      setProviderConfigured(true);
    } catch (err) {
      console.error("Provider configure failed:", err);
    } finally {
      setProviderSaving(false);
    }
  }, [selectedProvider, apiKey]);

  const handleCreateAgent = useCallback(async () => {
    if (selectedArchetype === null) return;
    const arch = AGENT_ARCHETYPES[selectedArchetype];
    setAgentCreating(true);
    try {
      await agents.register({
        name: arch.label,
        slug: arch.slug,
        description: arch.description,
        model: arch.model,
        system_prompt: arch.systemPrompt,
        max_steps: arch.maxSteps,
        temperature: arch.temperature,
      });
      setAgentSlug(arch.slug);
      setAgentCreated(true);
    } catch (err) {
      console.error("Agent create failed:", err);
    } finally {
      setAgentCreating(false);
    }
  }, [selectedArchetype]);

  const handleRunAgent = useCallback(async () => {
    if (!userMessage.trim() || !agentSlug) return;
    setRunning(true);
    setResponse(null);
    try {
      const res = await fetch("/api/v1/agents/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("aria_token") ?? ""}`,
        },
        body: JSON.stringify({ agent_slug: agentSlug, message: userMessage.trim() }),
      });
      const data = await res.json();
      setResponse(data.response ?? data.result ?? JSON.stringify(data, null, 2));
    } catch (err) {
      setResponse(`Error: ${err instanceof Error ? err.message : "Request failed"}`);
    } finally {
      setRunning(false);
    }
  }, [agentSlug, userMessage]);

  const handleComplete = () => {
    localStorage.setItem("onboarding_complete", "true");
    window.location.reload();
  };

  const canAdvance = (): boolean => {
    switch (step) {
      case 0: return true;
      case 1: return providerConfigured;
      case 2: return agentCreated;
      case 3: return true;
      default: return false;
    }
  };

  /* --- Render --- */

  return (
    <div
      className="flex min-h-screen items-center justify-center p-4 relative overflow-hidden"
      style={{ background: "#050510" }}
    >
      {/* Aurora background — matches login page */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          opacity: 0.8,
          background: [
            "radial-gradient(ellipse 70% 50% at 30% 50%, rgba(168, 85, 247, 0.25) 0%, transparent 50%)",
            "radial-gradient(ellipse 60% 40% at 70% 50%, rgba(0, 255, 170, 0.15) 0%, transparent 45%)",
            "radial-gradient(ellipse 40% 30% at 50% 30%, rgba(0, 212, 255, 0.12) 0%, transparent 40%)",
          ].join(", "),
        }}
      />
      {/* Scanlines */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          opacity: 0.03,
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.04) 2px, rgba(255,255,255,0.04) 4px)",
        }}
      />

      <div className="w-full max-w-xl relative z-10">
        {/* Step indicator */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {STEP_LABELS.map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <div
                className="flex items-center justify-center h-7 w-7 rounded-full text-[10px] font-semibold transition-all duration-300"
                style={{
                  background: i <= step ? "rgba(0, 255, 170, 0.15)" : "rgba(255, 255, 255, 0.04)",
                  border: i <= step ? "1px solid rgba(0, 255, 170, 0.4)" : "1px solid rgba(255, 255, 255, 0.08)",
                  color: i <= step ? "#00FFAA" : "rgba(255, 255, 255, 0.25)",
                  boxShadow: i === step ? "0 0 12px rgba(0, 255, 170, 0.2)" : "none",
                }}
              >
                {i + 1}
              </div>
              {i < STEP_LABELS.length - 1 && (
                <div
                  className="h-px w-8"
                  style={{
                    background: i < step
                      ? "rgba(0, 255, 170, 0.3)"
                      : "rgba(255, 255, 255, 0.06)",
                  }}
                />
              )}
            </div>
          ))}
        </div>

        {/* Card */}
        <div
          className="rounded-xl p-6 hud-fade-in"
          style={{
            background: "rgba(15, 20, 35, 0.7)",
            border: "1px solid rgba(0, 255, 170, 0.12)",
            backdropFilter: "blur(24px)",
            boxShadow:
              "0 0 0 1px rgba(0, 255, 170, 0.08), 0 8px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
          }}
        >
          {/* Step 1: Welcome */}
          {step === 0 && (
            <div className="text-center py-4">
              <img
                src="/aria-icon.png"
                alt="Aria Core"
                className="h-20 w-20 rounded-2xl object-cover mx-auto mb-6"
                style={{
                  boxShadow: "0 0 40px rgba(0, 255, 170, 0.4), 0 0 80px rgba(0, 255, 170, 0.15), 0 8px 32px rgba(0,0,0,0.4)",
                }}
              />
              <h2
                className="text-2xl font-semibold tracking-wide mb-2"
                style={{ color: "#fff", textShadow: "0 0 30px rgba(0, 255, 170, 0.2)" }}
              >
                Welcome to Aria Core
              </h2>
              <p className="text-sm mb-1" style={{ color: "rgba(255,255,255,0.5)" }}>
                Multi-tenant AI orchestration platform
              </p>
              <p className="text-xs font-mono" style={{ color: "rgba(255,255,255,0.25)" }}>
                Let&apos;s get your first agent running in 3 steps.
              </p>
            </div>
          )}

          {/* Step 2: Configure Provider */}
          {step === 1 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white mb-1">Configure a Provider</h3>
                <p className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
                  Paste an API key for at least one LLM provider.
                </p>
              </div>

              {/* Provider selector */}
              <div className="flex gap-2">
                {PROVIDER_OPTIONS.map((p) => (
                  <button
                    key={p.key}
                    onClick={() => { setSelectedProvider(p.key); setProviderConfigured(false); setApiKey(""); }}
                    className="flex-1 rounded-lg py-2 px-3 text-xs font-medium transition-all"
                    style={{
                      background: selectedProvider === p.key ? "rgba(0,255,170,0.1)" : "rgba(255,255,255,0.03)",
                      border: selectedProvider === p.key
                        ? "1px solid rgba(0,255,170,0.3)"
                        : "1px solid rgba(255,255,255,0.06)",
                      color: selectedProvider === p.key ? "#00FFAA" : "rgba(255,255,255,0.5)",
                    }}
                  >
                    {p.icon} {p.label}
                  </button>
                ))}
              </div>

              {/* API Key input */}
              <div>
                <label className="text-xs text-white/40 uppercase tracking-wider">API Key</label>
                <div className="relative mt-1">
                  <input
                    className="hud-input w-full pr-10 font-mono text-xs"
                    type={showKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={PROVIDER_OPTIONS.find((p) => p.key === selectedProvider)?.placeholder}
                    disabled={providerConfigured}
                  />
                  <button
                    type="button"
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 text-xs font-mono transition-colors"
                    onClick={() => setShowKey((v) => !v)}
                  >
                    {showKey ? "HIDE" : "SHOW"}
                  </button>
                </div>
              </div>

              {providerConfigured ? (
                <div
                  className="flex items-center gap-2 rounded-lg px-3 py-2"
                  style={{ background: "rgba(0,255,136,0.08)", border: "1px solid rgba(0,255,136,0.2)" }}
                >
                  <span style={{ color: "#00FF88" }}>&#10003;</span>
                  <span className="text-xs" style={{ color: "#00FF88" }}>
                    {PROVIDER_OPTIONS.find((p) => p.key === selectedProvider)?.label} configured
                  </span>
                </div>
              ) : (
                <button
                  className="hud-btn hud-btn--primary w-full"
                  onClick={handleConfigureProvider}
                  disabled={providerSaving || !apiKey.trim()}
                >
                  {providerSaving ? "Configuring..." : "Configure Provider"}
                </button>
              )}
            </div>
          )}

          {/* Step 3: Create Agent */}
          {step === 2 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white mb-1">Create Your First Agent</h3>
                <p className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
                  Choose an archetype to get started quickly.
                </p>
              </div>

              <div className="space-y-2 max-h-[300px] overflow-y-auto hud-scroll">
                {AGENT_ARCHETYPES.map((arch, i) => (
                  <button
                    key={arch.slug}
                    onClick={() => setSelectedArchetype(i)}
                    disabled={agentCreated}
                    className="w-full text-left rounded-lg p-3 transition-all"
                    style={{
                      background: selectedArchetype === i ? "rgba(0,255,170,0.06)" : "rgba(255,255,255,0.02)",
                      border: selectedArchetype === i
                        ? "1px solid rgba(0,255,170,0.25)"
                        : "1px solid rgba(255,255,255,0.06)",
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{arch.icon}</span>
                      <div>
                        <div className="text-sm font-medium text-white">{arch.label}</div>
                        <div className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>
                          {arch.model} — {arch.description}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              {agentCreated ? (
                <div
                  className="flex items-center gap-2 rounded-lg px-3 py-2"
                  style={{ background: "rgba(0,255,136,0.08)", border: "1px solid rgba(0,255,136,0.2)" }}
                >
                  <span style={{ color: "#00FF88" }}>&#10003;</span>
                  <span className="text-xs" style={{ color: "#00FF88" }}>
                    Agent &quot;{AGENT_ARCHETYPES[selectedArchetype!]?.label}&quot; created
                  </span>
                </div>
              ) : (
                <button
                  className="hud-btn hud-btn--primary w-full"
                  onClick={handleCreateAgent}
                  disabled={agentCreating || selectedArchetype === null}
                >
                  {agentCreating ? "Creating..." : "Create Agent"}
                </button>
              )}
            </div>
          )}

          {/* Step 4: Run Agent */}
          {step === 3 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white mb-1">Run Your First Agent</h3>
                <p className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
                  Send a message to see your agent in action.
                </p>
              </div>

              <div>
                <label className="text-xs text-white/40 uppercase tracking-wider">Message</label>
                <textarea
                  className="hud-input mt-1 w-full h-20 resize-none text-sm"
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  placeholder="e.g. Summarize the key benefits of AI orchestration..."
                />
              </div>

              <button
                className="hud-btn hud-btn--primary w-full"
                onClick={handleRunAgent}
                disabled={running || !userMessage.trim()}
              >
                {running ? "Running..." : "Run Agent"}
              </button>

              {response !== null && (
                <div
                  className="rounded-lg p-4 max-h-[200px] overflow-y-auto hud-scroll"
                  style={{
                    background: "rgba(0, 255, 170, 0.04)",
                    border: "1px solid rgba(0, 255, 170, 0.15)",
                  }}
                >
                  <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: "rgba(0,255,170,0.5)" }}>
                    Agent Response
                  </div>
                  <p className="text-xs font-mono whitespace-pre-wrap" style={{ color: "rgba(255,255,255,0.7)" }}>
                    {response}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Navigation buttons */}
          <div className="flex items-center justify-between mt-6 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <button
              className="hud-btn hud-btn--ghost text-xs"
              onClick={() => setStep((s) => s - 1)}
              disabled={step === 0}
              style={{ visibility: step === 0 ? "hidden" : "visible" }}
            >
              Back
            </button>

            <div className="flex gap-2">
              {step < 3 && (
                <button
                  className="hud-btn hud-btn--ghost text-xs"
                  onClick={handleComplete}
                >
                  Skip
                </button>
              )}
              {step < 3 ? (
                <button
                  className="hud-btn hud-btn--primary text-xs"
                  onClick={() => setStep((s) => s + 1)}
                  disabled={!canAdvance()}
                >
                  Next
                </button>
              ) : (
                <button
                  className="hud-btn hud-btn--primary text-xs"
                  onClick={handleComplete}
                >
                  Enter Dashboard
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-center mt-8">
          <div
            className="h-px w-32"
            style={{ background: "linear-gradient(90deg, transparent, rgba(0, 255, 170, 0.3), transparent)" }}
          />
        </div>
        <div className="text-center mt-3">
          <span className="text-[9px] font-mono" style={{ color: "rgba(255,255,255,0.15)" }}>
            Hidden Leaf Networks
          </span>
        </div>
      </div>
    </div>
  );
}

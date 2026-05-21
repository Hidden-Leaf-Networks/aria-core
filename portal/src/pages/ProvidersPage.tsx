import { useState, useCallback } from "react";
import { HudTopBar, HudPanel } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import {
  providers,
  type ProviderInfo,
  type ModelInfo,
} from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const PROVIDER_META: Record<string, { icon: string; label: string }> = {
  openai:    { icon: "\uD83D\uDFE2", label: "OpenAI" },
  anthropic: { icon: "\uD83D\uDFE3", label: "Anthropic" },
  xai:       { icon: "\u26A1",       label: "xAI" },
  google:    { icon: "\uD83D\uDD35", label: "Google" },
  local:     { icon: "\uD83D\uDDA5\uFE0F", label: "Local" },
};

const PROVIDER_KEYS = Object.keys(PROVIDER_META);

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export function ProvidersPage() {
  const { data: providerList, refetch: refetchProviders } = useApi(
    () => providers.list(),
    [],
    15000,
  );
  const { data: modelList } = useApi(() => providers.models(), [], 30000);
  const { data: status } = useApi(() => providers.status(), [], 15000);

  const [configuring, setConfiguring] = useState<string | null>(null);
  const [modelFilter, setModelFilter] = useState<string>("all");

  /* Build lookup by provider key */
  const providerMap: Record<string, ProviderInfo> = {};
  if (providerList) {
    for (const p of providerList) {
      providerMap[p.provider] = p;
    }
  }

  const filteredModels = modelList
    ? modelFilter === "all"
      ? modelList
      : modelList.filter((m) => m.provider === modelFilter)
    : [];

  return (
    <div>
      <HudTopBar
        title="Providers"
        subtitle="Configure API keys and models for each LLM provider"
        actions={
          <span className="text-xs font-mono text-white/40">
            {status
              ? `${status.configured_count} configured \u00B7 ${status.available_models} models`
              : "\u2014"}
          </span>
        }
      />

      <div className="p-6 space-y-8">
        {/* ---- Provider Cards ---- */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PROVIDER_KEYS.map((key) => {
            const info = providerMap[key];
            const connected = !!info?.has_key;
            const meta = PROVIDER_META[key];

            if (configuring === key) {
              return (
                <ConfigureForm
                  key={key}
                  providerKey={key}
                  meta={meta}
                  existing={info ?? null}
                  models={modelList?.filter((m) => m.provider === key) ?? []}
                  onDone={() => {
                    setConfiguring(null);
                    refetchProviders();
                  }}
                  onCancel={() => setConfiguring(null)}
                />
              );
            }

            return (
              <ProviderCard
                key={key}
                providerKey={key}
                meta={meta}
                info={info ?? null}
                connected={connected}
                onConfigure={() => setConfiguring(key)}
                onRemove={() => {
                  providers.remove(key).then(() => refetchProviders());
                }}
                onTest={() => {
                  providers.test(key).then((r) => {
                    alert(
                      r.status === "ok"
                        ? `${meta.label}: Connection OK`
                        : `${meta.label}: ${r.error ?? "Unknown error"}`,
                    );
                  });
                }}
              />
            );
          })}
        </div>

        {/* ---- Model Registry ---- */}
        <HudPanel title="Model Registry" subtitle="All known models across providers">
          {/* Filter tabs */}
          <div className="flex gap-1 mb-4 flex-wrap">
            <FilterTab
              active={modelFilter === "all"}
              label="All"
              onClick={() => setModelFilter("all")}
            />
            {PROVIDER_KEYS.map((k) => (
              <FilterTab
                key={k}
                active={modelFilter === k}
                label={PROVIDER_META[k].label}
                onClick={() => setModelFilter(k)}
              />
            ))}
          </div>

          {/* Table */}
          <div className="overflow-x-auto hud-scroll">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-[10px] text-white/30 uppercase tracking-wider border-b border-white/5">
                  <th className="pb-2 pr-4">Model ID</th>
                  <th className="pb-2 pr-4">Name</th>
                  <th className="pb-2 pr-4">Provider</th>
                  <th className="pb-2 pr-4 text-right">Context</th>
                  <th className="pb-2 pr-4 text-center">Vision</th>
                  <th className="pb-2 pr-4 text-center">Thinking</th>
                  <th className="pb-2 pr-4 text-right">In $/1M</th>
                  <th className="pb-2 text-right">Out $/1M</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.length === 0 && (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-white/20">
                      No models found
                    </td>
                  </tr>
                )}
                {filteredModels.map((m) => {
                  const available = providerMap[m.provider]?.has_key;
                  return (
                    <tr
                      key={m.id}
                      className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="py-2 pr-4 font-mono text-white/60">
                        {m.id}
                        {available && (
                          <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] bg-green-500/10 text-green-400">
                            Available
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-white/50">{m.name}</td>
                      <td className="py-2 pr-4 text-white/40">
                        {PROVIDER_META[m.provider]?.icon ?? ""}{" "}
                        {PROVIDER_META[m.provider]?.label ?? m.provider}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-white/40">
                        {(m.context_window / 1000).toFixed(0)}k
                      </td>
                      <td className="py-2 pr-4 text-center">
                        {m.supports_vision ? (
                          <span className="text-green-400">\u2713</span>
                        ) : (
                          <span className="text-white/15">\u2014</span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-center">
                        {m.supports_extended_thinking ? (
                          <span className="text-green-400">\u2713</span>
                        ) : (
                          <span className="text-white/15">\u2014</span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-white/30">
                        ${m.input_price_per_1m.toFixed(2)}
                      </td>
                      <td className="py-2 text-right font-mono text-white/30">
                        ${m.output_price_per_1m.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </HudPanel>

        {/* ---- Status Bar ---- */}
        <div className="text-center text-[10px] font-mono text-white/20">
          {status
            ? `${status.configured_count} provider${status.configured_count !== 1 ? "s" : ""} configured \u2014 ${status.available_models} model${status.available_models !== 1 ? "s" : ""} available`
            : "Loading provider status\u2026"}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Provider Card                                                      */
/* ------------------------------------------------------------------ */

function ProviderCard({
  providerKey,
  meta,
  info,
  connected,
  onConfigure,
  onRemove,
  onTest,
}: {
  providerKey: string;
  meta: { icon: string; label: string };
  info: ProviderInfo | null;
  connected: boolean;
  onConfigure: () => void;
  onRemove: () => void;
  onTest: () => void;
}) {
  return (
    <div
      className="rounded-xl p-5 backdrop-blur-md transition-all"
      style={{
        background: "rgba(255,255,255,0.02)",
        border: connected
          ? "1px solid rgba(0,255,170,0.25)"
          : "1px solid rgba(255,255,255,0.06)",
        boxShadow: connected ? "0 0 20px rgba(0,255,170,0.06)" : "none",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{meta.icon}</span>
          <div>
            <div className="text-sm font-medium text-white">{meta.label}</div>
            <div className="text-[10px] font-mono text-white/25">{providerKey}</div>
          </div>
        </div>
        <span
          className="text-[10px] font-mono px-2 py-0.5 rounded-full"
          style={{
            background: connected ? "rgba(0,255,170,0.1)" : "rgba(255,255,255,0.04)",
            color: connected ? "#00FFAA" : "rgba(255,255,255,0.3)",
          }}
        >
          {connected ? "Connected" : "Not Configured"}
        </span>
      </div>

      {info?.key_preview && (
        <div className="text-[10px] font-mono text-white/25 mb-2">
          Key: {info.key_preview}
        </div>
      )}

      {info?.default_model && (
        <div className="text-[10px] font-mono text-white/25 mb-3">
          Model: {info.default_model}
        </div>
      )}

      {info?.base_url && (
        <div className="text-[10px] font-mono text-white/20 mb-3 truncate">
          URL: {info.base_url}
        </div>
      )}

      <div className="flex gap-2 mt-3">
        <button className="hud-btn hud-btn--primary text-xs flex-1" onClick={onConfigure}>
          {connected ? "Reconfigure" : "Configure"}
        </button>
        {connected && (
          <>
            <button className="hud-btn hud-btn--secondary text-xs" onClick={onTest}>
              Test
            </button>
            <button className="hud-btn hud-btn--danger text-xs" onClick={onRemove}>
              Remove
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Configure Form                                                     */
/* ------------------------------------------------------------------ */

function ConfigureForm({
  providerKey,
  meta,
  existing,
  models,
  onDone,
  onCancel,
}: {
  providerKey: string;
  meta: { icon: string; label: string };
  existing: ProviderInfo | null;
  models: ModelInfo[];
  onDone: () => void;
  onCancel: () => void;
}) {
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [baseUrl, setBaseUrl] = useState(existing?.base_url ?? "");
  const [defaultModel, setDefaultModel] = useState(existing?.default_model ?? "");
  const [saving, setSaving] = useState(false);

  const handleSave = useCallback(async () => {
    if (!apiKey) return;
    setSaving(true);
    try {
      await providers.configure({
        provider: providerKey,
        api_key: apiKey,
        default_model: defaultModel || undefined,
        base_url: baseUrl || undefined,
      });
      onDone();
    } catch {
      setSaving(false);
    }
  }, [apiKey, baseUrl, defaultModel, providerKey, onDone]);

  return (
    <div
      className="rounded-xl p-5 backdrop-blur-md"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(0,255,170,0.2)",
        boxShadow: "0 0 30px rgba(0,255,170,0.05)",
      }}
    >
      <div className="flex items-center gap-3 mb-4">
        <span className="text-2xl">{meta.icon}</span>
        <div>
          <div className="text-sm font-medium text-white">Configure {meta.label}</div>
          <div className="text-[10px] font-mono text-white/25">{providerKey}</div>
        </div>
      </div>

      <div className="space-y-3">
        {/* API Key */}
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">API Key</label>
          <div className="relative mt-1">
            <input
              className="hud-input w-full pr-10"
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={existing?.key_preview ?? "sk-..."}
              autoFocus
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

        {/* Base URL */}
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">
            Base URL <span className="text-white/20">(optional)</span>
          </label>
          <input
            className="hud-input mt-1 w-full"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1"
          />
        </div>

        {/* Default Model */}
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Default Model</label>
          <select
            className="hud-select mt-1 w-full"
            value={defaultModel}
            onChange={(e) => setDefaultModel(e.target.value)}
          >
            <option value="">None</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.id})
              </option>
            ))}
          </select>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <button
            className="hud-btn hud-btn--primary flex-1"
            onClick={handleSave}
            disabled={saving || !apiKey}
          >
            {saving ? "Saving\u2026" : "Save"}
          </button>
          <button className="hud-btn hud-btn--secondary" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Filter Tab                                                         */
/* ------------------------------------------------------------------ */

function FilterTab({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className="px-3 py-1 rounded-md text-xs font-mono transition-all"
      style={{
        background: active ? "rgba(0,255,170,0.1)" : "rgba(255,255,255,0.03)",
        color: active ? "#00FFAA" : "rgba(255,255,255,0.35)",
        border: active
          ? "1px solid rgba(0,255,170,0.2)"
          : "1px solid rgba(255,255,255,0.05)",
      }}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

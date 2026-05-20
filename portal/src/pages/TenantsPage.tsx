import { useState } from "react";
import { HudTopBar, HudPanel, HudStatusPill, HudToggle } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { tenants, type Tenant, type TenantConfig } from "@/lib/api";

export function TenantsPage() {
  const { data: tenantList, refetch } = useApi(() => tenants.list(), []);
  const [selected, setSelected] = useState<Tenant | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div>
      <HudTopBar
        title="Tenants"
        subtitle="Manage white-label tenant deployments"
        actions={
          <button className="hud-btn hud-btn--primary" onClick={() => setShowCreate(true)}>
            + New Tenant
          </button>
        }
      />
      <div className="p-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_400px]">
          {/* Tenant list */}
          <HudPanel title="All Tenants" subtitle={`${tenantList?.length ?? 0} registered`}>
            {tenantList && tenantList.length > 0 ? (
              <div className="space-y-2">
                {tenantList.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setSelected(t)}
                    className={`w-full text-left rounded-lg border p-3 transition-all ${
                      selected?.id === t.id
                        ? "border-hud-accent/30 bg-white/[0.04]"
                        : "border-transparent hover:bg-white/[0.02]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium text-white">{t.name}</div>
                        <div className="text-xs text-white/40 font-mono">{t.slug}</div>
                      </div>
                      <HudStatusPill state={t.is_active ? "active" : "inactive"} />
                    </div>
                    <div className="mt-2 flex gap-4 text-xs text-white/30">
                      <span>Model: {t.config.default_model || "system default"}</span>
                      <span>Agents: {t.config.max_concurrent_agents}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/30 py-8 text-center">No tenants yet</p>
            )}
          </HudPanel>

          {/* Detail / Config editor */}
          {selected ? (
            <TenantDetail tenant={selected} onUpdate={refetch} />
          ) : showCreate ? (
            <CreateTenantForm
              onCreated={() => {
                setShowCreate(false);
                refetch();
              }}
              onCancel={() => setShowCreate(false)}
            />
          ) : (
            <HudPanel className="flex items-center justify-center min-h-[300px]">
              <p className="text-sm text-white/30">Select a tenant or create a new one</p>
            </HudPanel>
          )}
        </div>
      </div>
    </div>
  );
}

function TenantDetail({ tenant, onUpdate }: { tenant: Tenant; onUpdate: () => void }) {
  const [config, setConfig] = useState<Partial<TenantConfig>>(tenant.config);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    try {
      await tenants.updateConfig(tenant.id, config);
      onUpdate();
    } finally {
      setSaving(false);
    }
  };

  const updateFeature = (key: string, val: boolean) => {
    setConfig((prev) => ({
      ...prev,
      features: { ...prev.features, [key]: val },
    }));
  };

  return (
    <HudPanel
      title={tenant.name}
      subtitle={`ID: ${tenant.id.slice(0, 8)}...`}
      actions={
        <button className="hud-btn hud-btn--primary text-xs" onClick={save} disabled={saving}>
          {saving ? "Saving..." : "Save Config"}
        </button>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Slug</label>
          <div className="text-sm font-mono text-white/70 mt-1">{tenant.slug}</div>
        </div>

        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Default Model</label>
          <input
            className="hud-input mt-1"
            value={config.default_model || ""}
            onChange={(e) => setConfig({ ...config, default_model: e.target.value })}
            placeholder="e.g. claude-sonnet-4-20250514"
          />
        </div>

        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Max Concurrent Agents</label>
          <input
            className="hud-input mt-1"
            type="number"
            value={config.max_concurrent_agents ?? 10}
            onChange={(e) => setConfig({ ...config, max_concurrent_agents: Number(e.target.value) })}
          />
        </div>

        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Max Plans/Hour</label>
          <input
            className="hud-input mt-1"
            type="number"
            value={config.max_plans_per_hour ?? 100}
            onChange={(e) => setConfig({ ...config, max_plans_per_hour: Number(e.target.value) })}
          />
        </div>

        {/* Feature flags */}
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider mb-2 block">Feature Flags</label>
          <div className="space-y-2">
            {["deep_bridge", "planning", "voice", "event_sourcing"].map((flag) => (
              <HudToggle
                key={flag}
                label={flag.replace(/_/g, " ")}
                checked={config.features?.[flag] ?? false}
                onChange={(val) => updateFeature(flag, val)}
              />
            ))}
          </div>
        </div>

        <div className="pt-2 border-t border-white/5 text-xs text-white/20 font-mono">
          Created: {new Date(tenant.created_at).toLocaleString()}
        </div>
      </div>
    </HudPanel>
  );
}

function CreateTenantForm({
  onCreated,
  onCancel,
}: {
  onCreated: () => void;
  onCancel: () => void;
}) {
  const [slug, setSlug] = useState("");
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    if (!slug || !name) return;
    setSaving(true);
    try {
      await tenants.create({ slug, name });
      onCreated();
    } finally {
      setSaving(false);
    }
  };

  return (
    <HudPanel title="Create Tenant">
      <div className="space-y-4">
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Name</label>
          <input
            className="hud-input mt-1"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Acme Corporation"
          />
        </div>
        <div>
          <label className="text-xs text-white/40 uppercase tracking-wider">Slug</label>
          <input
            className="hud-input mt-1 font-mono"
            value={slug}
            onChange={(e) => setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ""))}
            placeholder="acme-corp"
          />
        </div>
        <div className="flex gap-2 pt-2">
          <button className="hud-btn hud-btn--primary flex-1" onClick={submit} disabled={saving}>
            {saving ? "Creating..." : "Create"}
          </button>
          <button className="hud-btn hud-btn--secondary" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </HudPanel>
  );
}

import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { useWebSocket } from "@/hooks/useWebSocket";
import { tenants, plans, approvals, events, agents, ws } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Metric card with icon + accent glow                                 */
/* ------------------------------------------------------------------ */

function MetricCard({
  label,
  value,
  icon,
  accent = "teal",
  onClick,
}: {
  label: string;
  value: string | number;
  icon: string;
  accent?: "teal" | "purple" | "cyan" | "warning" | "error";
  onClick?: () => void;
}) {
  const colors: Record<string, { glow: string; text: string; bar: string }> = {
    teal: { glow: "rgba(0,255,170,0.12)", text: "#00FFAA", bar: "#00FFAA" },
    purple: { glow: "rgba(168,85,247,0.12)", text: "#A855F7", bar: "#A855F7" },
    cyan: { glow: "rgba(0,212,255,0.12)", text: "#00D4FF", bar: "#00D4FF" },
    warning: { glow: "rgba(255,184,0,0.12)", text: "#FFB800", bar: "#FFB800" },
    error: { glow: "rgba(255,77,106,0.12)", text: "#FF4D6A", bar: "#FF4D6A" },
  };
  const c = colors[accent];

  return (
    <button
      onClick={onClick}
      className="rounded-xl p-4 text-left transition-all duration-300 hover:scale-[1.02] group"
      style={{
        background: "rgba(15, 20, 35, 0.7)",
        border: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(16px)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
      }}
    >
      {/* Accent bar */}
      <div className="h-0.5 w-10 rounded-full mb-3 transition-all duration-300 group-hover:w-full" style={{ background: c.bar }} />
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-[0.12em]" style={{ color: "rgba(255,255,255,0.35)" }}>
            {label}
          </div>
          <div className="text-2xl font-semibold mt-1 tabular-nums" style={{ color: c.text, textShadow: `0 0 20px ${c.glow}` }}>
            {value}
          </div>
        </div>
        <div className="text-2xl opacity-30 group-hover:opacity-60 transition-opacity">{icon}</div>
      </div>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Activity sparkline (simple bar chart)                               */
/* ------------------------------------------------------------------ */

function Sparkline({ data, color = "#00FFAA" }: { data: number[]; color?: string }) {
  const max = Math.max(...data, 1);
  return (
    <div className="flex items-end gap-px h-8">
      {data.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm min-w-[3px] transition-all duration-300"
          style={{
            height: `${(v / max) * 100}%`,
            background: `linear-gradient(180deg, ${color} 0%, ${color}44 100%)`,
            minHeight: v > 0 ? "2px" : "0px",
          }}
        />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main dashboard                                                      */
/* ------------------------------------------------------------------ */

export function DashboardPage() {
  const navigate = useNavigate();
  const { data: tenantList } = useApi(() => tenants.list(), [], 15000);
  const { data: planList } = useApi(() => plans.list({ limit: 10 }), [], 10000);
  const { data: approvalList } = useApi(() => approvals.list({ limit: 20 }), [], 10000);
  const { data: eventCount } = useApi(() => events.count(), [], 10000);
  const { data: recentEvents } = useApi(() => events.list({ limit: 8 }), [], 5000);
  const { data: agentList } = useApi(() => agents.list(), [], 15000);
  const { data: wsStatus } = useApi(() => ws.status(), [], 5000);
  const { connected, events: liveEvents } = useWebSocket(50);

  const activeTenants = tenantList?.filter((t) => t.is_active).length ?? 0;
  const pendingApprovals = approvalList?.filter((a) => a.state === "pending").length ?? 0;
  const completedPlans = planList?.filter((p) => p.state === "completed").length ?? 0;
  const failedPlans = planList?.filter((p) => p.state === "failed").length ?? 0;
  const totalPlans = planList?.length ?? 0;
  const successRate = totalPlans > 0 ? Math.round((completedPlans / totalPlans) * 100) : 100;

  // Fake sparkline data (in production this comes from event time-series)
  const activityData = useMemo(() => Array.from({ length: 24 }, () => Math.floor(Math.random() * 20)), []);
  const planActivityData = useMemo(() => Array.from({ length: 12 }, () => Math.floor(Math.random() * 8)), []);

  return (
    <div>
      <HudTopBar
        title="Dashboard"
        subtitle="Aria Core system overview"
        actions={
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className="h-2 w-2 rounded-full"
                style={{
                  background: connected ? "#00FF88" : "rgba(255,255,255,0.2)",
                  boxShadow: connected ? "0 0 8px rgba(0,255,136,0.6)" : "none",
                  animation: connected ? "pulse 2s infinite" : "none",
                }}
              />
              <span className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.35)" }}>
                {connected ? "LIVE" : "OFFLINE"}
              </span>
            </div>
            <div className="text-[10px] font-mono px-2 py-1 rounded" style={{ background: "rgba(0,255,170,0.08)", color: "rgba(0,255,170,0.6)" }}>
              v3.0.0
            </div>
          </div>
        }
      />

      <div className="p-6 space-y-6">
        {/* Metric cards — top row */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <MetricCard label="Active Tenants" value={activeTenants} icon="⬡" accent="teal" onClick={() => navigate("/tenants")} />
          <MetricCard label="Total Plans" value={totalPlans} icon="▶" accent="cyan" onClick={() => navigate("/plans")} />
          <MetricCard label="Pending Approvals" value={pendingApprovals} icon="✓" accent={pendingApprovals > 0 ? "warning" : "teal"} onClick={() => navigate("/approvals")} />
          <MetricCard label="Agents" value={agentList?.length ?? 0} icon="⬢" accent="purple" onClick={() => navigate("/agents")} />
          <MetricCard label="Events" value={eventCount?.count ?? 0} icon="◉" accent="cyan" onClick={() => navigate("/events")} />
        </div>

        {/* Two-column layout */}
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
          {/* Left column */}
          <div className="space-y-6">
            {/* System health strip */}
            <div className="grid grid-cols-3 gap-4">
              <HudPanel accent="teal">
                <div className="text-[10px] font-mono uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Success Rate</div>
                <div className="text-xl font-semibold mt-1" style={{ color: successRate >= 80 ? "#00FF88" : successRate >= 50 ? "#FFB800" : "#FF4D6A" }}>
                  {successRate}%
                </div>
                <div className="mt-2 h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${successRate}%`, background: successRate >= 80 ? "#00FF88" : "#FFB800" }} />
                </div>
              </HudPanel>

              <HudPanel accent="cyan">
                <div className="text-[10px] font-mono uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>WS Connections</div>
                <div className="text-xl font-semibold mt-1" style={{ color: "#00D4FF" }}>{wsStatus?.total_connections ?? 0}</div>
                <div className="mt-2">
                  <Sparkline data={activityData.slice(0, 12)} color="#00D4FF" />
                </div>
              </HudPanel>

              <HudPanel accent="purple">
                <div className="text-[10px] font-mono uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.3)" }}>Plan Activity</div>
                <div className="text-xl font-semibold mt-1" style={{ color: "#A855F7" }}>{totalPlans}</div>
                <div className="mt-2">
                  <Sparkline data={planActivityData} color="#A855F7" />
                </div>
              </HudPanel>
            </div>

            {/* Recent plans */}
            <HudPanel title="Recent Plans" subtitle={`${totalPlans} total — ${completedPlans} completed, ${failedPlans} failed`}>
              {planList && planList.length > 0 ? (
                <div className="space-y-2">
                  {planList.slice(0, 6).map((plan) => (
                    <div
                      key={plan.id}
                      className="flex items-center justify-between rounded-lg p-3 transition-all hover:bg-white/[0.02] cursor-pointer"
                      onClick={() => navigate("/plans")}
                      style={{ border: "1px solid rgba(255,255,255,0.03)" }}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="h-8 w-8 rounded-lg flex items-center justify-center text-xs font-bold"
                          style={{
                            background: plan.state === "completed" ? "rgba(0,255,136,0.1)" : plan.state === "failed" ? "rgba(255,77,106,0.1)" : "rgba(0,212,255,0.1)",
                            color: plan.state === "completed" ? "#00FF88" : plan.state === "failed" ? "#FF4D6A" : "#00D4FF",
                          }}
                        >
                          {plan.actions.length}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-white">{plan.name}</div>
                          <div className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.3)" }}>
                            {plan.created_by} — {new Date(plan.created_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      <HudStatusPill state={plan.state} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-8 text-center">
                  <div className="text-2xl mb-2 opacity-20">▶</div>
                  <p className="text-sm" style={{ color: "rgba(255,255,255,0.25)" }}>No plans yet</p>
                  <button className="hud-btn hud-btn--primary text-xs mt-3" onClick={() => navigate("/plans")}>Create Plan</button>
                </div>
              )}
            </HudPanel>
          </div>

          {/* Right column */}
          <div className="space-y-6">
            {/* Live event feed */}
            <HudPanel
              title="Live Feed"
              subtitle={connected ? `${liveEvents.length} events captured` : "Connecting..."}
              selected={connected}
              accent="teal"
            >
              <div className="space-y-1 max-h-[280px] overflow-y-auto hud-scroll">
                {liveEvents.length > 0 ? (
                  liveEvents.slice(0, 15).map((evt, i) => (
                    <div key={`${evt.timestamp}-${i}`} className="flex gap-2 py-1.5 text-[10px]" style={{ borderBottom: "1px solid rgba(255,255,255,0.02)" }}>
                      <span className="font-mono shrink-0 w-14" style={{ color: "rgba(255,255,255,0.2)" }}>
                        {new Date(evt.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                      <span className="font-mono truncate" style={{ color: "#00FFAA" }}>{evt.event_type}</span>
                    </div>
                  ))
                ) : recentEvents && recentEvents.length > 0 ? (
                  recentEvents.map((evt) => (
                    <div key={evt.id} className="flex gap-2 py-1.5 text-[10px]" style={{ borderBottom: "1px solid rgba(255,255,255,0.02)" }}>
                      <span className="font-mono shrink-0 w-14" style={{ color: "rgba(255,255,255,0.2)" }}>
                        {new Date(evt.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                      <span className="font-mono truncate" style={{ color: "#00FFAA" }}>{evt.event_type}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-[10px] py-4 text-center" style={{ color: "rgba(255,255,255,0.2)" }}>Waiting for events...</p>
                )}
              </div>
            </HudPanel>

            {/* Pending approvals */}
            <HudPanel
              title="Approval Queue"
              subtitle={`${pendingApprovals} pending`}
              accent={pendingApprovals > 0 ? "teal" : undefined}
            >
              {approvalList && approvalList.filter((a) => a.state === "pending").length > 0 ? (
                <div className="space-y-2">
                  {approvalList.filter((a) => a.state === "pending").slice(0, 4).map((a) => (
                    <div
                      key={a.id}
                      className="rounded-lg p-3 cursor-pointer hover:bg-white/[0.02] transition-all"
                      onClick={() => navigate("/approvals")}
                      style={{ border: "1px solid rgba(255,184,0,0.15)", background: "rgba(255,184,0,0.03)" }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-white">{a.gate_name}</span>
                        <span className="text-[10px] font-mono" style={{ color: "#FFB800" }}>Risk: {a.risk_score}</span>
                      </div>
                      <div className="text-[10px] font-mono mt-1" style={{ color: "rgba(255,255,255,0.25)" }}>
                        Expires {new Date(a.expires_at).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-4 text-center">
                  <div className="text-lg mb-1" style={{ color: "rgba(0,255,136,0.4)" }}>✓</div>
                  <p className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.25)" }}>All clear</p>
                </div>
              )}
            </HudPanel>

            {/* Quick actions */}
            <HudPanel title="Quick Actions">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: "New Plan", icon: "▶", to: "/plans" },
                  { label: "Register Agent", icon: "⬢", to: "/agents" },
                  { label: "Workflow Editor", icon: "◈", to: "/workflow" },
                  { label: "Marketplace", icon: "◎", to: "/marketplace" },
                ].map((action) => (
                  <button
                    key={action.to}
                    onClick={() => navigate(action.to)}
                    className="rounded-lg p-3 text-left transition-all duration-200 hover:scale-[1.02]"
                    style={{
                      background: "rgba(255,255,255,0.02)",
                      border: "1px solid rgba(255,255,255,0.05)",
                    }}
                  >
                    <div className="text-lg mb-1 opacity-40">{action.icon}</div>
                    <div className="text-xs font-medium" style={{ color: "rgba(255,255,255,0.6)" }}>{action.label}</div>
                  </button>
                ))}
              </div>
            </HudPanel>
          </div>
        </div>
      </div>
    </div>
  );
}

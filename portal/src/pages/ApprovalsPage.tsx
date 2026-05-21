import { useState } from "react";
import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { approvals } from "@/lib/api";

export function ApprovalsPage() {
  const { data: approvalList, refetch } = useApi(() => approvals.list({ limit: 100 }), [], 10000);
  const [acting, setActing] = useState<string | null>(null);

  const pending = (approvalList ?? []).filter((a) => a.state === "pending");
  const resolved = (approvalList ?? []).filter((a) => a.state !== "pending");

  const handleApprove = async (id: string) => {
    setActing(id);
    try { await approvals.approve(id); refetch(); } finally { setActing(null); }
  };

  const handleReject = async (id: string) => {
    setActing(id);
    try { await approvals.reject(id); refetch(); } finally { setActing(null); }
  };

  return (
    <div>
      <HudTopBar title="Approvals" subtitle={`${pending.length} pending`} />
      <div className="p-6 space-y-6">
        {/* Pending queue */}
        {pending.length > 0 && (
          <HudPanel title="Pending Queue" selected>
            <div className="space-y-3">
              {pending.map((a) => (
                <div key={a.id} className="flex items-center justify-between rounded-lg border border-hud-warning/20 bg-hud-warning/5 p-4">
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">{a.gate_name}</div>
                    <div className="text-xs text-white/40 font-mono mt-0.5">
                      Risk: {a.risk_score} — Requires {a.required_approvals} approval(s)
                    </div>
                    <div className="text-[10px] text-white/20 font-mono mt-1">
                      Expires: {new Date(a.expires_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <button
                      className="hud-btn hud-btn--primary text-xs"
                      onClick={() => handleApprove(a.id)}
                      disabled={acting === a.id}
                    >
                      {acting === a.id ? "..." : "Approve"}
                    </button>
                    <button
                      className="hud-btn hud-btn--danger text-xs"
                      onClick={() => handleReject(a.id)}
                      disabled={acting === a.id}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </HudPanel>
        )}

        {pending.length === 0 && (
          <HudPanel>
            <div className="py-6 text-center">
              <div className="text-2xl mb-2">✓</div>
              <div className="text-sm text-white/50">No pending approvals</div>
            </div>
          </HudPanel>
        )}

        {/* Resolved history */}
        {resolved.length > 0 && (
          <HudPanel title="Decision History" subtitle={`${resolved.length} resolved`}>
            <table className="hud-table">
              <thead>
                <tr>
                  <th>Gate</th>
                  <th>State</th>
                  <th>Risk</th>
                  <th>Created</th>
                  <th>Resolved</th>
                </tr>
              </thead>
              <tbody>
                {resolved.map((a) => (
                  <tr key={a.id}>
                    <td className="font-medium text-white">{a.gate_name}</td>
                    <td><HudStatusPill state={a.state} /></td>
                    <td className="font-mono text-white/60">{a.risk_score}</td>
                    <td className="text-white/40 text-xs font-mono">{new Date(a.created_at).toLocaleString()}</td>
                    <td className="text-white/40 text-xs font-mono">{new Date(a.expires_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </HudPanel>
        )}
      </div>
    </div>
  );
}

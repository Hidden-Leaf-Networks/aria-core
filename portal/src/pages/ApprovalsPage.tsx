import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { approvals } from "@/lib/api";

export function ApprovalsPage() {
  const { data: approvalList } = useApi(() => approvals.list({ limit: 100 }), [], 10000);

  const pending = (approvalList ?? []).filter((a) => a.state === "pending");

  return (
    <div>
      <HudTopBar
        title="Approvals"
        subtitle={`${pending.length} pending`}
      />
      <div className="p-6 space-y-6">
        {/* Pending queue */}
        {pending.length > 0 && (
          <HudPanel title="Pending Queue" selected>
            <div className="space-y-3">
              {pending.map((a) => (
                <div key={a.id} className="flex items-center justify-between rounded-lg border border-hud-warning/20 bg-hud-warning/5 p-3">
                  <div>
                    <div className="text-sm font-medium text-white">
                      {a.gate_name}
                    </div>
                    <div className="text-xs text-white/40 font-mono mt-0.5">
                      Risk: {a.risk_score} — Requires {a.required_approvals} approval(s)
                    </div>
                  </div>
                  <div className="text-xs text-white/30 font-mono">
                    Expires: {new Date(a.expires_at).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </HudPanel>
        )}

        {/* All approvals */}
        <HudPanel title="All Approvals">
          {approvalList && approvalList.length > 0 ? (
            <table className="hud-table">
              <thead>
                <tr>
                  <th>Gate</th>
                  <th>State</th>
                  <th>Risk Score</th>
                  <th>Created</th>
                  <th>Expires</th>
                </tr>
              </thead>
              <tbody>
                {approvalList.map((a) => (
                  <tr key={a.id}>
                    <td className="font-medium text-white">{a.gate_name}</td>
                    <td><HudStatusPill state={a.state} /></td>
                    <td className="font-mono text-white/60">{a.risk_score}</td>
                    <td className="text-white/40 text-xs font-mono">
                      {new Date(a.created_at).toLocaleString()}
                    </td>
                    <td className="text-white/40 text-xs font-mono">
                      {new Date(a.expires_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-white/30 py-8 text-center">No approvals</p>
          )}
        </HudPanel>
      </div>
    </div>
  );
}

import { HudTopBar, HudPanel, HudMetric } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { tenants, plans, approvals, events, ws } from "@/lib/api";

export function DashboardPage() {
  const { data: tenantList } = useApi(() => tenants.list(), [], 15000);
  const { data: planList } = useApi(() => plans.list({ limit: 5 }), [], 10000);
  const { data: approvalList } = useApi(() => approvals.list({ limit: 5 }), [], 10000);
  const { data: eventCount } = useApi(() => events.count(), [], 10000);
  const { data: wsStatus } = useApi(() => ws.status(), [], 5000);

  const activeTenants = tenantList?.filter((t) => t.is_active).length ?? 0;
  const pendingApprovals = approvalList?.filter((a) => a.state === "pending").length ?? 0;

  return (
    <div>
      <HudTopBar
        title="Dashboard"
        subtitle="Aria Core system overview"
      />
      <div className="p-6 space-y-6">
        {/* Metrics strip */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <HudPanel>
            <HudMetric label="Tenants" value={activeTenants} />
          </HudPanel>
          <HudPanel>
            <HudMetric label="Plans" value={planList?.length ?? 0} />
          </HudPanel>
          <HudPanel>
            <HudMetric
              label="Pending Approvals"
              value={pendingApprovals}
              trend={pendingApprovals > 0 ? "up" : "neutral"}
            />
          </HudPanel>
          <HudPanel>
            <HudMetric label="Total Events" value={eventCount?.count ?? 0} />
          </HudPanel>
          <HudPanel>
            <HudMetric label="WS Connections" value={wsStatus?.total_connections ?? 0} />
          </HudPanel>
        </div>

        {/* Recent plans */}
        <HudPanel title="Recent Plans" subtitle="Last 5 plans across all tenants">
          {planList && planList.length > 0 ? (
            <table className="hud-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>State</th>
                  <th>Actions</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {planList.map((plan) => (
                  <tr key={plan.id}>
                    <td className="font-medium text-white">{plan.name}</td>
                    <td>
                      <span className={`hud-pill hud-pill--${plan.state === "completed" ? "active" : plan.state === "failed" ? "error" : "info"}`}>
                        {plan.state}
                      </span>
                    </td>
                    <td className="text-white/60">{plan.actions.length}</td>
                    <td className="text-white/40 text-xs font-mono">
                      {new Date(plan.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-white/30 py-4 text-center">No plans yet</p>
          )}
        </HudPanel>
      </div>
    </div>
  );
}

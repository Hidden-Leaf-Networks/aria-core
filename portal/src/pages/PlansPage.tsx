import { HudTopBar, HudPanel, HudStatusPill } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { plans } from "@/lib/api";

export function PlansPage() {
  const { data: planList } = useApi(() => plans.list({ limit: 100 }), [], 10000);

  const byState = (planList ?? []).reduce<Record<string, number>>((acc, p) => {
    acc[p.state] = (acc[p.state] || 0) + 1;
    return acc;
  }, {});

  return (
    <div>
      <HudTopBar title="Plans" subtitle="Plan lifecycle management" />
      <div className="p-6 space-y-6">
        {/* State breakdown */}
        <div className="flex gap-3 flex-wrap">
          {Object.entries(byState).map(([state, count]) => (
            <div key={state} className="hud-panel flex items-center gap-2 px-3 py-2">
              <HudStatusPill state={state} />
              <span className="text-sm font-mono text-white/60">{count}</span>
            </div>
          ))}
        </div>

        {/* Plan table */}
        <HudPanel title="All Plans">
          {planList && planList.length > 0 ? (
            <table className="hud-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>State</th>
                  <th>Actions</th>
                  <th>Risk</th>
                  <th>Created By</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {planList.map((plan) => (
                  <tr key={plan.id}>
                    <td>
                      <div className="font-medium text-white">{plan.name}</div>
                      {plan.description && (
                        <div className="text-xs text-white/30 mt-0.5 truncate max-w-[200px]">
                          {plan.description}
                        </div>
                      )}
                    </td>
                    <td><HudStatusPill state={plan.state} /></td>
                    <td className="font-mono text-white/60">{plan.actions.length}</td>
                    <td className="font-mono text-white/60">
                      {plan.aggregate_risk_score ?? "—"}
                    </td>
                    <td className="text-white/40 text-xs">{plan.created_by}</td>
                    <td className="text-white/40 text-xs font-mono">
                      {new Date(plan.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-white/30 py-8 text-center">No plans</p>
          )}
        </HudPanel>
      </div>
    </div>
  );
}

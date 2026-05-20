import { useState } from "react";
import { HudTopBar, HudPanel, HudMetric } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { events, type AriaEvent } from "@/lib/api";

export function EventsPage() {
  const [filter, setFilter] = useState("");
  const { data: eventList } = useApi(
    () => events.list({ event_type: filter || undefined, limit: 200 }),
    [filter],
    5000
  );
  const { data: countData } = useApi(() => events.count(), [], 10000);
  const [replayData, setReplayData] = useState<AriaEvent[] | null>(null);
  const [replaying, setReplaying] = useState(false);

  const runReplay = async () => {
    setReplaying(true);
    try {
      const result = await events.replay({ event_type: filter || undefined });
      setReplayData(result.events);
    } finally {
      setReplaying(false);
    }
  };

  return (
    <div>
      <HudTopBar
        title="Events"
        subtitle="Append-only audit trail"
        actions={
          <button className="hud-btn hud-btn--secondary" onClick={runReplay} disabled={replaying}>
            {replaying ? "Replaying..." : "⟳ Replay"}
          </button>
        }
      />
      <div className="p-6 space-y-6">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="text-xs text-white/40 uppercase tracking-wider">Filter by type</label>
            <input
              className="hud-input mt-1"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="e.g. plan.created"
            />
          </div>
          <HudPanel className="shrink-0">
            <HudMetric label="Total Events" value={countData?.count ?? 0} />
          </HudPanel>
        </div>

        {/* Replay results */}
        {replayData && (
          <HudPanel
            title="Replay Results"
            subtitle={`${replayData.length} events (chronological)`}
            selected
            actions={
              <button className="hud-btn hud-btn--ghost text-xs" onClick={() => setReplayData(null)}>
                Close
              </button>
            }
          >
            <div className="max-h-[400px] overflow-y-auto hud-scroll space-y-1">
              {replayData.map((evt, i) => (
                <div key={evt.id} className="flex gap-3 py-1.5 text-xs border-b border-white/[0.03]">
                  <span className="text-white/20 font-mono w-8 shrink-0 text-right">{i + 1}</span>
                  <span className="text-hud-accent font-mono shrink-0 w-40">{evt.event_type}</span>
                  <span className="text-white/40 font-mono truncate">
                    {JSON.stringify(evt.payload)}
                  </span>
                </div>
              ))}
            </div>
          </HudPanel>
        )}

        {/* Event stream */}
        <HudPanel title="Recent Events" subtitle="Newest first">
          {eventList && eventList.length > 0 ? (
            <div className="max-h-[600px] overflow-y-auto hud-scroll">
              <table className="hud-table">
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Payload</th>
                    <th>Agent</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {eventList.map((evt) => (
                    <tr key={evt.id}>
                      <td className="font-mono text-hud-accent text-xs">{evt.event_type}</td>
                      <td className="text-white/40 text-xs font-mono max-w-[300px] truncate">
                        {JSON.stringify(evt.payload)}
                      </td>
                      <td className="text-white/30 text-xs font-mono">
                        {evt.agent_id ? evt.agent_id.slice(0, 8) : "—"}
                      </td>
                      <td className="text-white/30 text-xs font-mono">
                        {new Date(evt.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-white/30 py-8 text-center">No events</p>
          )}
        </HudPanel>
      </div>
    </div>
  );
}

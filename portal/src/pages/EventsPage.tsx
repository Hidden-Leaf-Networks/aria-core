import { useState } from "react";
import { HudTopBar, HudPanel, HudMetric } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { useWebSocket, type WSEvent } from "@/hooks/useWebSocket";
import { events, type AriaEvent } from "@/lib/api";

const EVENT_COLORS: Record<string, string> = {
  "plan.": "text-hud-info",
  "agent.": "text-hud-accent",
  "action.": "text-hud-warning",
  "approval.": "text-hud-accent2",
  "step.": "text-white/60",
  "transition.": "text-white/40",
};

function eventColor(type: string): string {
  for (const [prefix, color] of Object.entries(EVENT_COLORS)) {
    if (type.startsWith(prefix)) return color;
  }
  return "text-white/50";
}

export function EventsPage() {
  const [tab, setTab] = useState<"live" | "history" | "replay">("live");
  const [filter, setFilter] = useState("");
  const { events: liveEvents, connected, clear } = useWebSocket();
  const { data: eventList } = useApi(
    () => events.list({ event_type: filter || undefined, limit: 200 }),
    [filter],
    5000
  );
  const { data: countData } = useApi(() => events.count(), [], 10000);
  const [replayData, setReplayData] = useState<AriaEvent[] | null>(null);
  const [replaying, setReplaying] = useState(false);
  const [paused, setPaused] = useState(false);

  const runReplay = async () => {
    setReplaying(true);
    try {
      const result = await events.replay({ event_type: filter || undefined });
      setReplayData(result.events);
      setTab("replay");
    } finally {
      setReplaying(false);
    }
  };

  const displayedLive = paused ? [] : (filter
    ? liveEvents.filter((e) => e.event_type.includes(filter))
    : liveEvents);

  return (
    <div>
      <HudTopBar
        title="Events"
        subtitle="Append-only audit trail"
        actions={
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${connected ? "bg-hud-success animate-pulse" : "bg-white/20"}`} />
            <span className="text-xs text-white/40">{connected ? "Live" : "Disconnected"}</span>
          </div>
        }
      />
      <div className="p-6 space-y-4">
        {/* Controls */}
        <div className="flex gap-3 items-end flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <label className="text-xs text-white/40 uppercase tracking-wider">Filter by type</label>
            <input className="hud-input mt-1" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="e.g. plan.created" />
          </div>
          <HudPanel className="shrink-0">
            <HudMetric label="Total Events" value={countData?.count ?? 0} />
          </HudPanel>
          <div className="flex gap-1">
            {(["live", "history", "replay"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`hud-btn text-xs ${tab === t ? "hud-btn--primary" : "hud-btn--ghost"}`}
              >
                {t === "live" ? `Live (${liveEvents.length})` : t === "history" ? "History" : "Replay"}
              </button>
            ))}
          </div>
        </div>

        {/* Live stream */}
        {tab === "live" && (
          <HudPanel
            title="Live Event Stream"
            subtitle={connected ? "Streaming via WebSocket" : "Connecting..."}
            selected={connected}
            actions={
              <div className="flex gap-1">
                <button className={`hud-btn text-xs ${paused ? "hud-btn--primary" : "hud-btn--ghost"}`} onClick={() => setPaused(!paused)}>
                  {paused ? "▶ Resume" : "⏸ Pause"}
                </button>
                <button className="hud-btn hud-btn--ghost text-xs" onClick={clear}>Clear</button>
              </div>
            }
          >
            <div className="max-h-[500px] overflow-y-auto hud-scroll space-y-0.5">
              {displayedLive.length > 0 ? displayedLive.map((evt, i) => (
                <LiveEventRow key={`${evt.timestamp}-${i}`} event={evt} />
              )) : (
                <p className="text-sm text-white/20 py-8 text-center">
                  {paused ? "Paused — events accumulating in background" : "Waiting for events..."}
                </p>
              )}
            </div>
          </HudPanel>
        )}

        {/* History */}
        {tab === "history" && (
          <HudPanel
            title="Event History"
            subtitle="From persistent store (newest first)"
            actions={
              <button className="hud-btn hud-btn--secondary text-xs" onClick={runReplay} disabled={replaying}>
                {replaying ? "..." : "⟳ Replay"}
              </button>
            }
          >
            <div className="max-h-[500px] overflow-y-auto hud-scroll">
              {eventList && eventList.length > 0 ? (
                <table className="hud-table">
                  <thead>
                    <tr><th>Type</th><th>Payload</th><th>Agent</th><th>Time</th></tr>
                  </thead>
                  <tbody>
                    {eventList.map((evt) => (
                      <tr key={evt.id}>
                        <td className={`font-mono text-xs ${eventColor(evt.event_type)}`}>{evt.event_type}</td>
                        <td className="text-white/40 text-xs font-mono max-w-[300px] truncate">{JSON.stringify(evt.payload)}</td>
                        <td className="text-white/30 text-xs font-mono">{evt.agent_id?.slice(0, 8) ?? "—"}</td>
                        <td className="text-white/30 text-xs font-mono">{new Date(evt.timestamp).toLocaleTimeString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-sm text-white/30 py-8 text-center">No events</p>
              )}
            </div>
          </HudPanel>
        )}

        {/* Replay */}
        {tab === "replay" && (
          <HudPanel
            title="Event Replay"
            subtitle={replayData ? `${replayData.length} events (chronological)` : "Run a replay from the History tab"}
            selected={!!replayData}
          >
            {replayData ? (
              <div className="max-h-[500px] overflow-y-auto hud-scroll space-y-0.5">
                {replayData.map((evt, i) => (
                  <div key={evt.id} className="flex gap-3 py-1.5 text-xs border-b border-white/[0.03]">
                    <span className="text-white/15 font-mono w-8 shrink-0 text-right">{i + 1}</span>
                    <span className={`font-mono shrink-0 w-40 ${eventColor(evt.event_type)}`}>{evt.event_type}</span>
                    <span className="text-white/30 font-mono truncate flex-1">{JSON.stringify(evt.payload)}</span>
                    <span className="text-white/15 font-mono shrink-0">{new Date(evt.timestamp).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/20 py-8 text-center">No replay data — run replay from History tab</p>
            )}
          </HudPanel>
        )}
      </div>
    </div>
  );
}

function LiveEventRow({ event }: { event: WSEvent }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="flex gap-3 py-1.5 px-2 rounded hover:bg-white/[0.02] cursor-pointer text-xs border-b border-white/[0.02]"
      onClick={() => setExpanded(!expanded)}
    >
      <span className="text-white/15 font-mono shrink-0 w-20">{new Date(event.timestamp).toLocaleTimeString()}</span>
      <span className={`font-mono shrink-0 w-40 ${eventColor(event.event_type)}`}>{event.event_type}</span>
      <span className="text-white/30 font-mono truncate flex-1">
        {expanded ? JSON.stringify(event.payload, null, 2) : JSON.stringify(event.payload)}
      </span>
    </div>
  );
}

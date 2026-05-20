import { useState } from "react";
import { HudTopBar, HudPanel } from "@/components/hud";
import { useApi } from "@/hooks/useApi";
import { contexts, type AgentContext } from "@/lib/api";

export function ContextsPage() {
  const { data: contextList } = useApi(() => contexts.list({ limit: 50 }), [], 10000);
  const [selected, setSelected] = useState<AgentContext | null>(null);

  return (
    <div>
      <HudTopBar title="Contexts" subtitle="Agent execution contexts" />
      <div className="p-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_420px]">
          <HudPanel title="Recent Contexts" subtitle={`${contextList?.length ?? 0} loaded`}>
            {contextList && contextList.length > 0 ? (
              <div className="space-y-2">
                {contextList.map((ctx) => (
                  <button
                    key={ctx.id}
                    onClick={() => setSelected(ctx)}
                    className={`w-full text-left rounded-lg border p-3 transition-all ${
                      selected?.id === ctx.id
                        ? "border-hud-accent3/30 bg-white/[0.04]"
                        : "border-transparent hover:bg-white/[0.02]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-white/60">{ctx.id.slice(0, 12)}...</span>
                      <span className="text-xs text-white/30">{ctx.messages.length} msgs</span>
                    </div>
                    <div className="text-xs text-white/30 font-mono mt-1">
                      Conv: {ctx.conversation_id.slice(0, 8)} — Steps: {ctx.step_count}
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/30 py-8 text-center">No contexts</p>
            )}
          </HudPanel>

          {selected ? (
            <HudPanel title="Context Inspector" subtitle={`ID: ${selected.id.slice(0, 12)}...`}>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-white/40 uppercase tracking-wider">Conversation</label>
                  <div className="text-xs font-mono text-white/60 mt-1">{selected.conversation_id}</div>
                </div>
                <div>
                  <label className="text-xs text-white/40 uppercase tracking-wider">Steps</label>
                  <div className="text-sm text-white mt-1">{selected.step_count}</div>
                </div>
                <div>
                  <label className="text-xs text-white/40 uppercase tracking-wider mb-2 block">
                    Messages ({selected.messages.length})
                  </label>
                  <div className="space-y-2 max-h-[400px] overflow-y-auto hud-scroll">
                    {selected.messages.map((msg, i) => (
                      <div
                        key={i}
                        className={`rounded-lg p-2 text-xs ${
                          msg.role === "user"
                            ? "bg-hud-info/10 border border-hud-info/20"
                            : msg.role === "assistant"
                              ? "bg-hud-accent/5 border border-hud-accent/10"
                              : "bg-white/5 border border-white/5"
                        }`}
                      >
                        <div className="font-mono text-[10px] text-white/30 mb-1 uppercase">{msg.role}</div>
                        <div className="text-white/70 whitespace-pre-wrap">{msg.content}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </HudPanel>
          ) : (
            <HudPanel className="flex items-center justify-center min-h-[300px]">
              <p className="text-sm text-white/30">Select a context to inspect</p>
            </HudPanel>
          )}
        </div>
      </div>
    </div>
  );
}

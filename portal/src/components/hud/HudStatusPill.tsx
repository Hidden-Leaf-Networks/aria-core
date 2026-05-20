import { clsx } from "clsx";

const STATE_MAP: Record<string, string> = {
  active: "hud-pill--active",
  completed: "hud-pill--active",
  approved: "hud-pill--active",
  draft: "hud-pill--info",
  planned: "hud-pill--info",
  queued: "hud-pill--info",
  pending: "hud-pill--warning",
  executing: "hud-pill--warning",
  blocked: "hud-pill--warning",
  failed: "hud-pill--error",
  rejected: "hud-pill--error",
  expired: "hud-pill--error",
  error: "hud-pill--error",
  inactive: "hud-pill--inactive",
  archived: "hud-pill--inactive",
};

interface HudStatusPillProps {
  state: string;
  className?: string;
}

export function HudStatusPill({ state, className }: HudStatusPillProps) {
  const variant = STATE_MAP[state.toLowerCase()] || "hud-pill--inactive";
  return (
    <span className={clsx("hud-pill", variant, className)}>
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
      {state}
    </span>
  );
}

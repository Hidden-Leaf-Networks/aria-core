import { clsx } from "clsx";

interface HudMetricProps {
  label: string;
  value: string | number;
  trend?: "up" | "down" | "neutral";
  className?: string;
}

export function HudMetric({ label, value, trend, className }: HudMetricProps) {
  return (
    <div className={clsx("hud-metric", className)}>
      <span className="hud-metric-label">{label}</span>
      <span
        className={clsx(
          "hud-metric-value",
          trend === "up" && "text-hud-success",
          trend === "down" && "text-hud-error",
          !trend && "text-white"
        )}
      >
        {value}
      </span>
    </div>
  );
}

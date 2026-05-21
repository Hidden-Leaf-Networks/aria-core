import { clsx } from "clsx";

type SkeletonVariant = "text" | "card" | "table" | "metric";

interface HudSkeletonProps {
  variant?: SkeletonVariant;
  /** Number of rows for "table" variant, lines for "text" variant */
  rows?: number;
  className?: string;
}

function SkeletonText({ rows = 3, className }: { rows: number; className?: string }) {
  return (
    <div className={clsx("space-y-3", className)}>
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className="hud-skeleton rounded"
          style={{
            height: 14,
            width: i === rows - 1 ? "45%" : i % 2 === 0 ? "100%" : "80%",
          }}
        />
      ))}
    </div>
  );
}

function SkeletonCard({ className }: { className?: string }) {
  return (
    <div
      className={clsx("rounded-xl p-5", className)}
      style={{
        background: "rgba(15, 20, 35, 0.7)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
        backdropFilter: "blur(16px)",
      }}
    >
      <div className="hud-skeleton rounded mb-4" style={{ height: 20, width: "40%" }} />
      <div className="space-y-3">
        <div className="hud-skeleton rounded" style={{ height: 14, width: "100%" }} />
        <div className="hud-skeleton rounded" style={{ height: 14, width: "75%" }} />
        <div className="hud-skeleton rounded" style={{ height: 14, width: "60%" }} />
      </div>
    </div>
  );
}

function SkeletonTable({ rows = 5, className }: { rows: number; className?: string }) {
  return (
    <div
      className={clsx("rounded-xl overflow-hidden", className)}
      style={{
        background: "rgba(15, 20, 35, 0.7)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
      }}
    >
      {/* Header */}
      <div
        className="flex gap-4 px-4 py-3"
        style={{ background: "rgba(0, 0, 0, 0.2)", borderBottom: "1px solid var(--hud-border)" }}
      >
        {[30, 25, 20, 15].map((w, i) => (
          <div key={i} className="hud-skeleton rounded" style={{ height: 10, width: `${w}%` }} />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className="flex gap-4 px-4 py-3"
          style={{ borderBottom: i < rows - 1 ? "1px solid rgba(255,255,255,0.03)" : "none" }}
        >
          {[30, 25, 20, 15].map((w, j) => (
            <div key={j} className="hud-skeleton rounded" style={{ height: 14, width: `${w}%` }} />
          ))}
        </div>
      ))}
    </div>
  );
}

function SkeletonMetric({ className }: { className?: string }) {
  return (
    <div
      className={clsx("rounded-lg p-4 relative overflow-hidden", className)}
      style={{
        background: "var(--hud-surface-glass)",
        backdropFilter: "blur(16px)",
        border: "1px solid var(--hud-border)",
        boxShadow: "var(--hud-panel-shadow)",
      }}
    >
      {/* Top gradient bar */}
      <div
        className="absolute top-0 left-0 right-0"
        style={{
          height: 2,
          background: "linear-gradient(90deg, var(--hud-accent), var(--hud-accent3), var(--hud-accent2))",
          opacity: 0.3,
        }}
      />
      <div className="hud-skeleton rounded mb-2" style={{ height: 10, width: "50%" }} />
      <div className="hud-skeleton rounded" style={{ height: 24, width: "65%" }} />
    </div>
  );
}

export function HudSkeleton({ variant = "text", rows = 3, className }: HudSkeletonProps) {
  switch (variant) {
    case "card":
      return <SkeletonCard className={className} />;
    case "table":
      return <SkeletonTable rows={rows} className={className} />;
    case "metric":
      return <SkeletonMetric className={className} />;
    case "text":
    default:
      return <SkeletonText rows={rows} className={className} />;
  }
}

import { type ReactNode } from "react";

interface HudTopBarProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

export function HudTopBar({ title, subtitle, actions }: HudTopBarProps) {
  return (
    <header
      className="sticky top-0 z-20 flex items-center justify-between px-6 py-4"
      style={{
        background: "rgba(5, 5, 16, 0.75)",
        borderBottom: "1px solid rgba(0, 255, 170, 0.08)",
        backdropFilter: "blur(20px)",
        boxShadow: "0 4px 24px rgba(0, 0, 0, 0.4)",
      }}
    >
      <div>
        <h1
          className="text-lg font-semibold tracking-wide"
          style={{
            color: "#fff",
            textShadow: "0 0 20px rgba(0, 255, 170, 0.15)",
          }}
        >
          {title}
        </h1>
        {subtitle && (
          <p className="text-xs font-mono mt-0.5" style={{ color: "rgba(255,255,255,0.4)" }}>
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </header>
  );
}

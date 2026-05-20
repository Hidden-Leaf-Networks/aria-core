import { type ReactNode } from "react";

interface HudTopBarProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

export function HudTopBar({ title, subtitle, actions }: HudTopBarProps) {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-white/[0.06] bg-hud-bg-dark/80 px-6 py-4 backdrop-blur-md">
      <div>
        <h1 className="text-lg font-semibold text-white">{title}</h1>
        {subtitle && (
          <p className="text-sm text-white/50">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

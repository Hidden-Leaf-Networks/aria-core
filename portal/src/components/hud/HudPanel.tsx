import { clsx } from "clsx";
import { type ReactNode } from "react";

interface HudPanelProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  selected?: boolean;
  className?: string;
}

export function HudPanel({
  title,
  subtitle,
  actions,
  children,
  selected,
  className,
}: HudPanelProps) {
  return (
    <div className={clsx("hud-panel", selected && "hud-panel--selected", className)}>
      {(title || actions) && (
        <div className="flex items-center justify-between mb-3">
          <div>
            {title && <h3 className="text-sm font-medium text-white">{title}</h3>}
            {subtitle && <p className="text-xs text-white/40 mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

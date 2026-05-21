import { clsx } from "clsx";
import { type ReactNode } from "react";

interface HudPanelProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  selected?: boolean;
  accent?: "teal" | "purple" | "cyan";
  className?: string;
}

const ACCENT_BORDERS: Record<string, string> = {
  teal: "rgba(0, 255, 170, 0.25)",
  purple: "rgba(168, 85, 247, 0.25)",
  cyan: "rgba(0, 212, 255, 0.25)",
};

const ACCENT_GLOWS: Record<string, string> = {
  teal: "0 0 20px rgba(0, 255, 170, 0.08)",
  purple: "0 0 20px rgba(168, 85, 247, 0.08)",
  cyan: "0 0 20px rgba(0, 212, 255, 0.08)",
};

export function HudPanel({
  title,
  subtitle,
  actions,
  children,
  selected,
  accent,
  className,
}: HudPanelProps) {
  const borderColor = selected
    ? "rgba(0, 255, 170, 0.3)"
    : accent
      ? ACCENT_BORDERS[accent]
      : "rgba(255, 255, 255, 0.06)";

  const shadow = selected
    ? "0 0 0 1px rgba(0, 255, 170, 0.3), 0 0 25px rgba(0, 255, 170, 0.1), 0 4px 24px rgba(0,0,0,0.5)"
    : accent
      ? `${ACCENT_GLOWS[accent]}, 0 4px 24px rgba(0,0,0,0.4)`
      : "0 0 0 1px rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)";

  return (
    <div
      className={clsx("rounded-xl p-5 transition-all duration-300", className)}
      style={{
        background: "rgba(15, 20, 35, 0.7)",
        border: `1px solid ${borderColor}`,
        backdropFilter: "blur(16px)",
        boxShadow: shadow,
      }}
    >
      {(title || actions) && (
        <div className="flex items-center justify-between mb-4">
          <div>
            {title && (
              <h3
                className="text-sm font-semibold tracking-wide"
                style={{ color: "rgba(255,255,255,0.92)" }}
              >
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-[11px] mt-0.5 font-mono" style={{ color: "rgba(255,255,255,0.35)" }}>
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

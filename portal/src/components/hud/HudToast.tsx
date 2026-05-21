import { useEffect, useState } from "react";
import type { ToastType } from "@/contexts/ToastContext";

interface HudToastProps {
  message: string;
  type: ToastType;
  onDismiss: () => void;
}

const TOAST_COLORS: Record<ToastType, { accent: string; bg: string; border: string; glow: string }> = {
  success: {
    accent: "#00FF88",
    bg: "rgba(0, 255, 136, 0.08)",
    border: "rgba(0, 255, 136, 0.25)",
    glow: "0 0 20px rgba(0, 255, 136, 0.15)",
  },
  error: {
    accent: "#FF4D6A",
    bg: "rgba(255, 77, 106, 0.08)",
    border: "rgba(255, 77, 106, 0.25)",
    glow: "0 0 20px rgba(255, 77, 106, 0.15)",
  },
  info: {
    accent: "#00B4FF",
    bg: "rgba(0, 180, 255, 0.08)",
    border: "rgba(0, 180, 255, 0.25)",
    glow: "0 0 20px rgba(0, 180, 255, 0.15)",
  },
};

const TOAST_ICONS: Record<ToastType, string> = {
  success: "M9 12l2 2 4-4",
  error: "M12 8v4m0 4h.01",
  info: "M12 16v-4m0-4h.01",
};

export function HudToast({ message, type, onDismiss }: HudToastProps) {
  const [visible, setVisible] = useState(false);
  const colors = TOAST_COLORS[type];

  useEffect(() => {
    // Trigger entrance animation
    const frame = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <div
      className="pointer-events-auto flex items-start gap-3 rounded-lg px-4 py-3 transition-all duration-300"
      style={{
        background: "rgba(15, 20, 35, 0.92)",
        backdropFilter: "blur(16px)",
        border: `1px solid ${colors.border}`,
        boxShadow: `${colors.glow}, 0 4px 24px rgba(0,0,0,0.5)`,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateX(0)" : "translateX(20px)",
      }}
    >
      {/* Icon */}
      <div
        className="flex-shrink-0 mt-0.5 flex h-5 w-5 items-center justify-center rounded-full"
        style={{ background: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke={colors.accent}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d={TOAST_ICONS[type]} />
        </svg>
      </div>

      {/* Message */}
      <p className="text-xs leading-relaxed flex-1" style={{ color: "rgba(255,255,255,0.85)" }}>
        {message}
      </p>

      {/* Dismiss */}
      <button
        onClick={onDismiss}
        className="flex-shrink-0 mt-0.5 text-white/20 hover:text-white/60 transition-colors"
        aria-label="Dismiss notification"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

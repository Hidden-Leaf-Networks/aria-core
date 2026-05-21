import { type ReactNode } from "react";

interface HudShellProps {
  nav: ReactNode;
  children: ReactNode;
}

export function HudShell({ nav, children }: HudShellProps) {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#050510" }}>
      {/* Aurora mesh — layered purple/teal/cyan radial gradients */}
      <div
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          opacity: 0.7,
          background: [
            "radial-gradient(ellipse 80% 50% at 15% 35%, rgba(168, 85, 247, 0.2) 0%, transparent 50%)",
            "radial-gradient(ellipse 60% 40% at 80% 60%, rgba(0, 255, 170, 0.13) 0%, transparent 45%)",
            "radial-gradient(ellipse 50% 30% at 50% 15%, rgba(0, 212, 255, 0.11) 0%, transparent 40%)",
          ].join(", "),
        }}
      />
      {/* Scanline texture */}
      <div
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          opacity: 0.025,
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.04) 2px, rgba(255,255,255,0.04) 4px)",
        }}
      />

      {/* Left nav — glass surface with glow edge */}
      <aside
        className="hidden w-[248px] shrink-0 lg:flex flex-col z-10 relative"
        style={{
          background: "rgba(10, 13, 28, 0.88)",
          borderRight: "1px solid rgba(0, 255, 170, 0.1)",
          backdropFilter: "blur(24px)",
          boxShadow: "4px 0 30px rgba(0, 0, 0, 0.5), inset -1px 0 0 rgba(255,255,255,0.03)",
        }}
      >
        <div className="flex h-full flex-col overflow-y-auto hud-scroll">{nav}</div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto hud-scroll z-10 relative">{children}</main>
    </div>
  );
}

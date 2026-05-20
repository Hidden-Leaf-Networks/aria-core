import { type ReactNode } from "react";

interface HudShellProps {
  nav: ReactNode;
  children: ReactNode;
}

export function HudShell({ nav, children }: HudShellProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-hud-bg-darkest">
      {/* Left nav rail */}
      <aside className="hidden w-60 shrink-0 border-r border-white/[0.06] bg-hud-bg-dark lg:block">
        <div className="flex h-full flex-col overflow-y-auto hud-scroll">
          {nav}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto hud-scroll">
        {children}
      </main>
    </div>
  );
}

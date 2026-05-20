import { NavLink } from "react-router-dom";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "◆" },
  { to: "/tenants", label: "Tenants", icon: "⬡" },
  { to: "/plans", label: "Plans", icon: "▶" },
  { to: "/approvals", label: "Approvals", icon: "✓" },
  { to: "/events", label: "Events", icon: "◉" },
  { to: "/contexts", label: "Contexts", icon: "◫" },
];

export function HudNav() {
  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-hud-accent to-hud-accent3 flex items-center justify-center text-sm font-bold text-hud-bg-darkest">
            A
          </div>
          <div>
            <div className="text-sm font-semibold text-white">Aria Core</div>
            <div className="text-[10px] text-white/40 font-mono">CONFIG PORTAL</div>
          </div>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              clsx("hud-nav-item", isActive && "hud-nav-item--active")
            }
          >
            <span className="text-base w-5 text-center opacity-60">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/[0.06]">
        <div className="text-[10px] text-white/30 font-mono">
          v0.3.0 — White-Label
        </div>
      </div>
    </div>
  );
}

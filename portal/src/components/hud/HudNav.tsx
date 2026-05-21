import { NavLink } from "react-router-dom";
import { clsx } from "clsx";
import { useAuth } from "@/contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "◆" },
  { to: "/tenants", label: "Tenants", icon: "⬡" },
  { to: "/plans", label: "Plans", icon: "▶" },
  { to: "/approvals", label: "Approvals", icon: "✓" },
  { to: "/events", label: "Events", icon: "◉" },
  { to: "/contexts", label: "Contexts", icon: "◫" },
  { to: "/agents", label: "Agents", icon: "⬢" },
  { to: "/workflow", label: "Workflow", icon: "◈" },
  { to: "/marketplace", label: "Marketplace", icon: "◎" },
];

export function HudNav() {
  const { logout } = useAuth();

  return (
    <div className="flex flex-col h-full">
      {/* Logo with glow */}
      <div className="px-5 py-5" style={{ borderBottom: "1px solid rgba(0, 255, 170, 0.08)" }}>
        <div className="flex items-center gap-3">
          <div
            className="h-9 w-9 rounded-xl flex items-center justify-center text-sm font-bold"
            style={{
              background: "linear-gradient(135deg, #00FFAA 0%, #00D4FF 50%, #A855F7 100%)",
              color: "#050510",
              boxShadow: "0 0 20px rgba(0, 255, 170, 0.35), 0 0 40px rgba(0, 255, 170, 0.15)",
            }}
          >
            A
          </div>
          <div>
            <div className="text-sm font-semibold text-white tracking-wide">Aria Core</div>
            <div
              className="text-[9px] font-mono tracking-[0.2em] uppercase"
              style={{ color: "rgba(0, 255, 170, 0.5)" }}
            >
              CONFIG PORTAL
            </div>
          </div>
        </div>
      </div>

      {/* Section label */}
      <div className="px-5 pt-5 pb-2">
        <div className="text-[9px] font-mono uppercase tracking-[0.15em]" style={{ color: "rgba(255,255,255,0.25)" }}>
          Navigation
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-3 space-y-0.5">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              clsx(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-200 relative",
                isActive
                  ? "text-white"
                  : "text-white/50 hover:text-white/80 hover:bg-white/[0.03]"
              )
            }
            style={({ isActive }) =>
              isActive
                ? {
                    background: "rgba(0, 255, 170, 0.08)",
                    borderLeft: "3px solid #00FFAA",
                    boxShadow: "inset 0 0 20px rgba(0, 255, 170, 0.05)",
                  }
                : {}
            }
          >
            <span
              className="text-base w-5 text-center transition-all duration-200"
              style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 170, 0.3))" }}
            >
              {item.icon}
            </span>
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Connection status indicator */}
      <div className="px-5 py-3" style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
        <div className="flex items-center gap-2 mb-3">
          <div
            className="h-2 w-2 rounded-full animate-pulse"
            style={{ background: "#00FF88", boxShadow: "0 0 8px rgba(0, 255, 136, 0.6)" }}
          />
          <span className="text-[10px] text-white/30 font-mono">System Online</span>
        </div>
        <button
          onClick={logout}
          className="w-full rounded-lg px-3 py-2 text-xs text-white/30 hover:text-red-400 hover:bg-red-500/5 transition-all duration-200 font-mono"
        >
          ↩ Disconnect
        </button>
        <div className="text-[9px] text-white/15 font-mono text-center mt-2 tracking-wider">
          v3.0.0-alpha.1
        </div>
      </div>
    </div>
  );
}

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

export function LoginPage() {
  const { login } = useAuth();
  const [token, setToken] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (token.trim()) login(token.trim());
  };

  return (
    <div
      className="flex min-h-screen items-center justify-center p-4 relative overflow-hidden"
      style={{ background: "#050510" }}
    >
      {/* Aurora background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          opacity: 0.8,
          background: [
            "radial-gradient(ellipse 70% 50% at 30% 50%, rgba(168, 85, 247, 0.25) 0%, transparent 50%)",
            "radial-gradient(ellipse 60% 40% at 70% 50%, rgba(0, 255, 170, 0.15) 0%, transparent 45%)",
            "radial-gradient(ellipse 40% 30% at 50% 30%, rgba(0, 212, 255, 0.12) 0%, transparent 40%)",
          ].join(", "),
        }}
      />
      {/* Scanlines */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          opacity: 0.03,
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.04) 2px, rgba(255,255,255,0.04) 4px)",
        }}
      />

      <div className="w-full max-w-sm relative z-10">
        {/* Logo with glow */}
        <div className="flex flex-col items-center gap-4 mb-10">
          <img
            src="/aria-icon.png"
            alt="Aria Core"
            className="h-20 w-20 rounded-2xl object-cover"
            style={{
              boxShadow: "0 0 40px rgba(0, 255, 170, 0.4), 0 0 80px rgba(0, 255, 170, 0.15), 0 8px 32px rgba(0,0,0,0.4)",
            }}
          />
          <div className="text-center">
            <div
              className="text-2xl font-semibold tracking-wide"
              style={{ color: "#fff", textShadow: "0 0 30px rgba(0, 255, 170, 0.2)" }}
            >
              Aria Core
            </div>
            <div
              className="text-[10px] font-mono tracking-[0.3em] uppercase mt-1"
              style={{ color: "rgba(0, 255, 170, 0.5)" }}
            >
              Config Portal
            </div>
          </div>
        </div>

        <form
          onSubmit={submit}
          className="rounded-xl p-6 space-y-5"
          style={{
            background: "rgba(15, 20, 35, 0.7)",
            border: "1px solid rgba(0, 255, 170, 0.12)",
            backdropFilter: "blur(24px)",
            boxShadow:
              "0 0 0 1px rgba(0, 255, 170, 0.08), 0 8px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
          }}
        >
          <div>
            <label
              className="text-[10px] font-mono uppercase tracking-[0.15em] block mb-2"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Authentication Token
            </label>
            <textarea
              className="hud-input font-mono text-xs h-28 resize-none"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste your JWT token..."
              autoFocus
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            />
          </div>
          <button
            type="submit"
            disabled={!token.trim()}
            className="w-full rounded-lg py-3 text-sm font-semibold transition-all duration-300 disabled:opacity-30"
            style={{
              background: token.trim()
                ? "linear-gradient(135deg, #00BFA5 0%, #00FFAA 100%)"
                : "rgba(255,255,255,0.05)",
              color: token.trim() ? "#050510" : "rgba(255,255,255,0.3)",
              boxShadow: token.trim()
                ? "0 0 20px rgba(0, 255, 170, 0.3), 0 4px 16px rgba(0,0,0,0.3)"
                : "none",
            }}
          >
            Authenticate
          </button>
          <p className="text-[10px] text-center font-mono" style={{ color: "rgba(255,255,255,0.2)" }}>
            Generate a token via the Aria Core API or CLI
          </p>
        </form>

        {/* Decorative bottom line */}
        <div className="flex justify-center mt-8">
          <div
            className="h-px w-32"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(0, 255, 170, 0.3), transparent)",
            }}
          />
        </div>
        <div className="text-center mt-3">
          <span className="text-[9px] font-mono" style={{ color: "rgba(255,255,255,0.15)" }}>
            Hidden Leaf Networks
          </span>
        </div>
      </div>
    </div>
  );
}

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
    <div className="flex min-h-screen items-center justify-center bg-hud-bg-darkest p-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-hud-accent to-hud-accent3 flex items-center justify-center text-xl font-bold text-hud-bg-darkest">
            A
          </div>
          <div>
            <div className="text-xl font-semibold text-white">Aria Core</div>
            <div className="text-xs text-white/40 font-mono uppercase tracking-wider">Config Portal</div>
          </div>
        </div>

        <form onSubmit={submit} className="hud-panel space-y-4">
          <div>
            <label className="text-xs text-white/40 uppercase tracking-wider">JWT Token</label>
            <textarea
              className="hud-input mt-1 font-mono text-xs h-24 resize-none"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste your JWT token here..."
              autoFocus
            />
          </div>
          <button type="submit" className="hud-btn hud-btn--primary w-full" disabled={!token.trim()}>
            Authenticate
          </button>
          <p className="text-[10px] text-white/20 text-center font-mono">
            Generate a token via the Aria Core API or CLI
          </p>
        </form>
      </div>
    </div>
  );
}

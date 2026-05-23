import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";

interface RegisterResponse {
  tenant_id: string;
  user_id: string;
  token: string;
  tier: string;
  message: string;
  error?: string;
}

export function SignupPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [company, setCompany] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isValid = name.trim() && email.trim() && password.length >= 8;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid) return;

    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/v1/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim(),
          password,
          company: company.trim() || undefined,
        }),
      });

      const data: RegisterResponse = await res.json();

      if (!res.ok) {
        setError(data.error || "Registration failed");
        return;
      }

      // Save token and redirect to dashboard
      login(data.token);
      navigate("/", { replace: true });
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
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
              Create Your Account
            </div>
          </div>
        </div>

        <form
          onSubmit={submit}
          className="rounded-xl p-6 space-y-4"
          style={{
            background: "rgba(15, 20, 35, 0.7)",
            border: "1px solid rgba(0, 255, 170, 0.12)",
            backdropFilter: "blur(24px)",
            boxShadow:
              "0 0 0 1px rgba(0, 255, 170, 0.08), 0 8px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
          }}
        >
          {error && (
            <div
              className="text-xs font-mono rounded-lg px-3 py-2"
              style={{
                background: "rgba(255, 60, 60, 0.1)",
                border: "1px solid rgba(255, 60, 60, 0.2)",
                color: "rgba(255, 120, 120, 0.9)",
              }}
            >
              {error}
            </div>
          )}

          <div>
            <label
              className="text-[10px] font-mono uppercase tracking-[0.15em] block mb-2"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Name
            </label>
            <input
              type="text"
              className="hud-input font-mono text-xs w-full rounded-lg px-3 py-2.5"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              autoFocus
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#fff",
              }}
            />
          </div>

          <div>
            <label
              className="text-[10px] font-mono uppercase tracking-[0.15em] block mb-2"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Email
            </label>
            <input
              type="email"
              className="hud-input font-mono text-xs w-full rounded-lg px-3 py-2.5"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#fff",
              }}
            />
          </div>

          <div>
            <label
              className="text-[10px] font-mono uppercase tracking-[0.15em] block mb-2"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Password
            </label>
            <input
              type="password"
              className="hud-input font-mono text-xs w-full rounded-lg px-3 py-2.5"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Min 8 characters"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#fff",
              }}
            />
          </div>

          <div>
            <label
              className="text-[10px] font-mono uppercase tracking-[0.15em] block mb-2"
              style={{ color: "rgba(255,255,255,0.35)" }}
            >
              Company <span style={{ color: "rgba(255,255,255,0.2)" }}>(optional)</span>
            </label>
            <input
              type="text"
              className="hud-input font-mono text-xs w-full rounded-lg px-3 py-2.5"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              placeholder="Your company"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#fff",
              }}
            />
          </div>

          <button
            type="submit"
            disabled={!isValid || loading}
            className="w-full rounded-lg py-3 text-sm font-semibold transition-all duration-300 disabled:opacity-30"
            style={{
              background: isValid
                ? "linear-gradient(135deg, #00BFA5 0%, #00FFAA 100%)"
                : "rgba(255,255,255,0.05)",
              color: isValid ? "#050510" : "rgba(255,255,255,0.3)",
              boxShadow: isValid
                ? "0 0 20px rgba(0, 255, 170, 0.3), 0 4px 16px rgba(0,0,0,0.3)"
                : "none",
            }}
          >
            {loading ? "Creating Account..." : "Create Account"}
          </button>

          <p className="text-[10px] text-center font-mono" style={{ color: "rgba(255,255,255,0.3)" }}>
            Already have an account?{" "}
            <Link
              to="/login"
              className="underline transition-colors"
              style={{ color: "rgba(0, 255, 170, 0.6)" }}
            >
              Login
            </Link>
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

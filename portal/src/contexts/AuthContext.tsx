import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { setToken } from "@/lib/api";

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setAuthToken] = useState<string | null>(() => {
    const saved = localStorage.getItem("aria_token");
    if (saved) setToken(saved);
    return saved;
  });

  const login = useCallback((newToken: string) => {
    const clean = newToken.replace(/\s+/g, "");
    localStorage.setItem("aria_token", clean);
    setToken(clean);
    setAuthToken(clean);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("aria_token");
    setToken("");
    setAuthToken(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        token,
        isAuthenticated: !!token,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { ToastProvider } from "@/contexts/ToastContext";
import { HudShell, HudNav, HudErrorBoundary } from "@/components/hud";
import { DashboardPage } from "@/pages/DashboardPage";
import { TenantsPage } from "@/pages/TenantsPage";
import { PlansPage } from "@/pages/PlansPage";
import { ApprovalsPage } from "@/pages/ApprovalsPage";
import { EventsPage } from "@/pages/EventsPage";
import { ContextsPage } from "@/pages/ContextsPage";
import { AgentsPage } from "@/pages/AgentsPage";
import { WorkflowEditorPage } from "@/pages/WorkflowEditorPage";
import { ProvidersPage } from "@/pages/ProvidersPage";
import { MarketplacePage } from "@/pages/MarketplacePage";
import { LoginPage } from "@/pages/LoginPage";
import { OnboardingPage } from "@/pages/OnboardingPage";
import { providers } from "@/lib/api";

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <LoginPage />;
  return <>{children}</>;
}

function OnboardingGate({ children }: { children: React.ReactNode }) {
  const [checking, setChecking] = useState(true);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    // Skip if already completed onboarding
    if (localStorage.getItem("onboarding_complete") === "true") {
      setChecking(false);
      return;
    }

    // Check if any providers are configured
    providers
      .status()
      .then((status) => {
        if (status.configured_count === 0) {
          setNeedsOnboarding(true);
        }
      })
      .catch(() => {
        // If API fails, skip onboarding (may be a first-time setup issue)
      })
      .finally(() => setChecking(false));
  }, []);

  if (checking) return null;
  if (needsOnboarding) return <OnboardingPage />;
  return <>{children}</>;
}

function AppLayout() {
  return (
    <HudShell nav={<HudNav />}>
      <HudErrorBoundary>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/tenants" element={<TenantsPage />} />
          <Route path="/plans" element={<PlansPage />} />
          <Route path="/approvals" element={<ApprovalsPage />} />
          <Route path="/events" element={<EventsPage />} />
          <Route path="/contexts" element={<ContextsPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/providers" element={<ProvidersPage />} />
          <Route path="/workflow" element={<WorkflowEditorPage />} />
          <Route path="/marketplace" element={<MarketplacePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HudErrorBoundary>
    </HudShell>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ToastProvider>
          <AuthGate>
            <OnboardingGate>
              <AppLayout />
            </OnboardingGate>
          </AuthGate>
        </ToastProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}

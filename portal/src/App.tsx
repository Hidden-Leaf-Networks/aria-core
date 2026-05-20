import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { HudShell, HudNav } from "@/components/hud";
import { DashboardPage } from "@/pages/DashboardPage";
import { TenantsPage } from "@/pages/TenantsPage";
import { PlansPage } from "@/pages/PlansPage";
import { ApprovalsPage } from "@/pages/ApprovalsPage";
import { EventsPage } from "@/pages/EventsPage";
import { ContextsPage } from "@/pages/ContextsPage";
import { LoginPage } from "@/pages/LoginPage";

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <LoginPage />;
  return <>{children}</>;
}

function AppLayout() {
  return (
    <HudShell nav={<HudNav />}>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/tenants" element={<TenantsPage />} />
        <Route path="/plans" element={<PlansPage />} />
        <Route path="/approvals" element={<ApprovalsPage />} />
        <Route path="/events" element={<EventsPage />} />
        <Route path="/contexts" element={<ContextsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HudShell>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AuthGate>
          <AppLayout />
        </AuthGate>
      </AuthProvider>
    </BrowserRouter>
  );
}

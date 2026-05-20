/**
 * Aria Core API client — typed fetch wrapper with auth.
 */

const BASE_URL = "/api/v1";

let _token: string | null = null;

export function setToken(token: string) {
  _token = token;
}

export function getToken(): string | null {
  return _token;
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (_token) {
    headers["Authorization"] = `Bearer ${_token}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.error || body.detail || res.statusText);
  }

  return res.json();
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// --- Tenants ---

export interface Tenant {
  id: string;
  slug: string;
  name: string;
  config: TenantConfig;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface TenantConfig {
  display_name?: string;
  logo_url?: string;
  default_model?: string;
  allowed_models: string[];
  max_tokens?: number;
  max_concurrent_agents: number;
  max_plans_per_hour: number;
  max_events_per_day: number;
  features: Record<string, boolean>;
  risk_policy_id?: string;
  metadata: Record<string, unknown>;
}

export const tenants = {
  list: () => request<Tenant[]>("/tenants"),
  get: (id: string) => request<Tenant>(`/tenants/${id}`),
  create: (data: { slug: string; name: string; config?: Partial<TenantConfig> }) =>
    request<Tenant>("/tenants", { method: "POST", body: JSON.stringify(data) }),
  updateConfig: (id: string, config: Partial<TenantConfig>) =>
    request<Tenant>(`/tenants/${id}/config`, { method: "PUT", body: JSON.stringify(config) }),
};

// --- Plans ---

export interface PlanAction {
  id: string;
  plan_id: string;
  index: number;
  name: string;
  description: string;
  skill_name?: string;
  state: string;
  error?: string;
  execution_time_ms?: number;
}

export interface Plan {
  id: string;
  name: string;
  description: string;
  state: string;
  actions: PlanAction[];
  current_action_index: number;
  aggregate_risk_score?: number;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export const plans = {
  list: (params?: { state?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    if (params?.state) qs.set("state", params.state);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    const q = qs.toString();
    return request<Plan[]>(`/plans${q ? `?${q}` : ""}`);
  },
  get: (id: string) => request<Plan>(`/plans/${id}`),
  create: (data: { name: string; description?: string; actions: unknown[] }) =>
    request<Plan>("/plans", { method: "POST", body: JSON.stringify(data) }),
  delete: (id: string) =>
    request<{ deleted: boolean }>(`/plans/${id}`, { method: "DELETE" }),
};

// --- Approvals ---

export interface Approval {
  id: string;
  plan_id: string;
  gate_name: string;
  risk_score: number;
  state: string;
  required_approvals: number;
  created_at: string;
  expires_at: string;
}

export const approvals = {
  list: (params?: { state?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.state) qs.set("state", params.state);
    if (params?.limit) qs.set("limit", String(params.limit));
    const q = qs.toString();
    return request<Approval[]>(`/approvals${q ? `?${q}` : ""}`);
  },
  get: (id: string) => request<Approval>(`/approvals/${id}`),
};

// --- Events ---

export interface AriaEvent {
  id: string;
  tenant_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  agent_id?: string;
  timestamp: string;
}

export const events = {
  list: (params?: { event_type?: string; agent_id?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.event_type) qs.set("event_type", params.event_type);
    if (params?.agent_id) qs.set("agent_id", params.agent_id);
    if (params?.limit) qs.set("limit", String(params.limit));
    const q = qs.toString();
    return request<AriaEvent[]>(`/events${q ? `?${q}` : ""}`);
  },
  replay: (params?: { event_type?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.event_type) qs.set("event_type", params.event_type);
    if (params?.limit) qs.set("limit", String(params.limit));
    const q = qs.toString();
    return request<{ count: number; events: AriaEvent[] }>(`/events/replay${q ? `?${q}` : ""}`);
  },
  count: (event_type?: string) => {
    const q = event_type ? `?event_type=${event_type}` : "";
    return request<{ count: number }>(`/events/count${q}`);
  },
};

// --- Contexts ---

export interface AgentContext {
  id: string;
  tenant_id: string;
  conversation_id: string;
  messages: { role: string; content: string }[];
  step_count: number;
  created_at: string;
}

export const contexts = {
  list: (params?: { conversation_id?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.conversation_id) qs.set("conversation_id", params.conversation_id);
    if (params?.limit) qs.set("limit", String(params.limit));
    const q = qs.toString();
    return request<AgentContext[]>(`/contexts${q ? `?${q}` : ""}`);
  },
  get: (id: string) => request<AgentContext>(`/contexts/${id}`),
};

// --- WebSocket Status ---

export const ws = {
  status: () => request<{ tenant_connections: number; total_connections: number }>("/ws/status"),
};

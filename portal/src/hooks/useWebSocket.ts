import { useState, useEffect, useRef, useCallback } from "react";
import { getToken } from "@/lib/api";

export interface WSEvent {
  event_type: string;
  payload: Record<string, unknown>;
  timestamp: string;
}

interface UseWebSocketResult {
  events: WSEvent[];
  connected: boolean;
  error: string | null;
  clear: () => void;
}

export function useWebSocket(maxEvents: number = 200): UseWebSocketResult {
  const [events, setEvents] = useState<WSEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const token = getToken();
    if (!token) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/events`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ token: `Bearer ${token}` }));
    };

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as WSEvent;
        if (data.event_type === "connected") {
          setConnected(true);
          setError(null);
          return;
        }
        setEvents((prev) => [data, ...prev].slice(0, maxEvents));
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => setError("WebSocket error");
    ws.onclose = () => { setConnected(false); wsRef.current = null; };

    return () => { ws.close(); };
  }, [maxEvents]);

  const clear = useCallback(() => setEvents([]), []);

  return { events, connected, error, clear };
}

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class HudErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[HudErrorBoundary]", error, info.componentStack);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex items-center justify-center min-h-[300px] p-6">
          <div
            className="rounded-xl p-6 max-w-lg w-full text-center"
            style={{
              background: "rgba(15, 20, 35, 0.85)",
              border: "1px solid rgba(255, 77, 106, 0.3)",
              backdropFilter: "blur(16px)",
              boxShadow:
                "0 0 0 1px rgba(255, 77, 106, 0.15), 0 0 30px rgba(255, 77, 106, 0.1), 0 4px 24px rgba(0,0,0,0.5)",
            }}
          >
            {/* Icon */}
            <div
              className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full"
              style={{
                background: "rgba(255, 77, 106, 0.12)",
                border: "1px solid rgba(255, 77, 106, 0.2)",
              }}
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#FF4D6A"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>

            <h3
              className="text-sm font-semibold tracking-wide mb-2"
              style={{ color: "#FF4D6A" }}
            >
              System Error
            </h3>

            <p
              className="text-xs font-mono mb-4 px-4 py-2 rounded-md text-left break-all"
              style={{
                background: "rgba(255, 77, 106, 0.06)",
                color: "rgba(255, 255, 255, 0.6)",
                border: "1px solid rgba(255, 77, 106, 0.1)",
              }}
            >
              {this.state.error?.message ?? "An unexpected error occurred"}
            </p>

            <button
              className="hud-btn hud-btn--danger text-sm px-6"
              onClick={this.handleRetry}
            >
              Retry
            </button>

            <p
              className="text-[10px] font-mono mt-4"
              style={{ color: "rgba(255, 255, 255, 0.2)" }}
            >
              If this persists, check the console for details
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

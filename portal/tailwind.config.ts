import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Aria Core HUD palette
        hud: {
          bg: {
            darkest: "#070A0F",
            dark: "#0B1220",
            medium: "#101A2E",
            light: "#1A2540",
          },
          accent: "#00FFAA",
          accent2: "#A855F7",
          accent3: "#00D4FF",
          primary: "#00BFA5",
          "primary-light": "#33CCBB",
          "primary-dark": "#009688",
          success: "#00C853",
          warning: "#FFB300",
          error: "#FF5252",
          info: "#00B0FF",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "monospace"],
        sans: ['"Inter"', "system-ui", "sans-serif"],
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        glow: "0 0 20px rgba(0, 255, 170, 0.15)",
        "glow-accent": "0 0 30px rgba(0, 255, 170, 0.25)",
        "glow-purple": "0 0 20px rgba(168, 85, 247, 0.15)",
      },
    },
  },
  plugins: [],
};

export default config;

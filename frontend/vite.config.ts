import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// DeepWeather AI (SIH26078) - Vite configuration.
// The dashboard talks to the FastAPI backend directly through
// VITE_API_BASE_URL (CORS is enabled server side for the dev origins).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  preview: {
    port: 4173,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
});

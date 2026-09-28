/// <reference types="vite/client" />

// Type-safe access to the Vite environment variables used by the dashboard.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

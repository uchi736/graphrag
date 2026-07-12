import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// 本番は FastAPI が dist を同一オリジンで配信するため base 相対。
// 開発時は Vite dev server から imagegen バックエンド(8100)へ各APIをプロキシ。
const backend = process.env.IMAGEGEN_BACKEND ?? "http://127.0.0.1:8100"
const apiPaths = ["/generate", "/edit", "/jobs", "/models", "/images", "/health"]

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist" },
  server: {
    proxy: Object.fromEntries(
      apiPaths.map((p) => [p, { target: backend, changeOrigin: true }]),
    ),
  },
})

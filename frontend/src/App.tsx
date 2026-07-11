import { lazy, Suspense } from "react"
import { Navigate, Route, Routes } from "react-router-dom"
import { AppHeader } from "@/components/layout/AppHeader"
import { TabNav } from "@/components/layout/TabNav"
import { HealthGate } from "@/components/layout/HealthGate"

// ページ単位で lazy 分割（グラフ可視化ライブラリをQA画面のバンドルから隔離）
const QaPage = lazy(() => import("@/pages/QaPage"))
const GraphPage = lazy(() => import("@/pages/GraphPage"))
const DocumentsPage = lazy(() => import("@/pages/DocumentsPage"))
const BuildPage = lazy(() => import("@/pages/BuildPage"))
const SettingsPage = lazy(() => import("@/pages/SettingsPage"))

function PageLoading() {
  return (
    <div className="py-16 text-center text-sm text-muted-foreground">
      読み込み中…
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen">
      <AppHeader />
      <TabNav />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <HealthGate>
          <Suspense fallback={<PageLoading />}>
            <Routes>
              <Route path="/" element={<Navigate to="/qa" replace />} />
              <Route path="/qa" element={<QaPage />} />
              <Route path="/graph" element={<GraphPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/build" element={<BuildPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<Navigate to="/qa" replace />} />
            </Routes>
          </Suspense>
        </HealthGate>
      </main>
      <footer className="border-t py-4 text-center text-xs text-muted-foreground">
        GraphRAG · Powered by LangChain, Neo4j &amp; PGVector
      </footer>
    </div>
  )
}

import { NavLink } from "react-router-dom"
import {
  MessageSquare,
  Network,
  FileText,
  Hammer,
  Settings,
} from "lucide-react"
import { cn } from "@/lib/utils"

const tabs = [
  { to: "/qa", label: "質問応答", icon: MessageSquare },
  { to: "/graph", label: "グラフ探索", icon: Network },
  { to: "/documents", label: "登録ドキュメント", icon: FileText },
  { to: "/build", label: "構築 / 取り込み", icon: Hammer },
  { to: "/settings", label: "設定", icon: Settings },
]

export function TabNav() {
  return (
    <nav className="border-b bg-card">
      <div className="mx-auto flex max-w-6xl gap-1 px-4">
        {tabs.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-1.5 border-b-2 px-4 py-3 text-sm font-medium transition-colors",
                isActive
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}

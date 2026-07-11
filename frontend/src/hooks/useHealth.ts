import { useQuery } from "@tanstack/react-query"
import { apiGet } from "@/api/client"
import type { HealthReport, SettingsInfo } from "@/api/types"

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGet<HealthReport>("/api/health"),
    staleTime: 30_000,
    retry: 1,
  })
}

export function useSettingsInfo() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: () => apiGet<SettingsInfo>("/api/settings"),
    staleTime: Infinity,
  })
}

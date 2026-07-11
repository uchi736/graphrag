import { useQaStream } from "@/hooks/useQaStream"
import { useSettingsStore, selectQaConfig } from "@/stores/settingsStore"
import { SearchSettingsPanel } from "@/components/qa/SearchSettingsPanel"
import { QuestionForm } from "@/components/qa/QuestionForm"
import { AnswerCard } from "@/components/qa/AnswerCard"
import { EvidencePanels } from "@/components/qa/EvidencePanels"
import { QaHistoryList } from "@/components/qa/QaHistoryList"

export default function QaPage() {
  const qa = useQaStream()
  const settings = useSettingsStore()

  return (
    <div className="space-y-4">
      <SearchSettingsPanel />
      <QuestionForm
        busy={qa.busy}
        onSubmit={(q) => qa.submit(q, selectQaConfig(settings))}
        onCancel={qa.cancel}
      />
      <AnswerCard state={qa} />
      <EvidencePanels evidence={qa.evidence} />
      <QaHistoryList />
    </div>
  )
}

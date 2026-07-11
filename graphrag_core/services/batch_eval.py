"""CSVバッチ評価サービス（scripts/batch_eval.py のUIジョブ版）。

question 列を持つCSVを受け取り、1行ずつ QA パイプライン（検索+生成）を実行して
answer / sources / kg_used 列を追記したCSV文字列を返す。入力の他の列
（expected 等）はそのまま結果に引き継ぐので、正解列を並べて目視採点できる。
"""
from __future__ import annotations

import csv
import io
import time
from typing import Callable, Dict, List, Optional

from graphrag_core.services.progress import JobCancelled, ProgressEvent, ProgressFn

_RESULT_COLS = ("answer", "sources", "kg_used", "n_graph_relations", "elapsed_sec")


def parse_questions_csv(data: bytes) -> List[Dict[str, str]]:
    """CSVバイト列を行dictリストへ。question 列必須、空行スキップ。"""
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "question" not in reader.fieldnames:
        raise ValueError("CSVに question 列が必要です（他の列は結果にそのまま引き継ぎます）")
    return [r for r in reader if (r.get("question") or "").strip()]


def run_batch_eval(rows: List[Dict], deps, config: Dict, *,
                   progress: Optional[ProgressFn] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """全質問を実行し {"n": 件数, "n_errors": 失敗数, "csv": 結果CSV文字列} を返す。"""
    from graphrag_core.services.qa import answer_question, serialize_qa_result

    out_rows: List[Dict] = []
    n_errors = 0
    for i, row in enumerate(rows, 1):
        if should_cancel and should_cancel():
            raise JobCancelled()
        q = (row.get("question") or "").strip()
        if progress:
            progress(ProgressEvent(stage="eval", current=i, total=len(rows),
                                   ok=len(out_rows) - n_errors, err=n_errors,
                                   message=f"{q[:42]}..."))
        t0 = time.time()
        try:
            r = serialize_qa_result(answer_question(q, deps, dict(config)))
            srcs = sorted({s.get("source")
                           for s in (r.get("vector_sources") or []) + (r.get("kg_source_chunks") or [])
                           if s.get("source")})
            out_rows.append({**row,
                             "answer": r.get("answer", ""),
                             "sources": "; ".join(srcs),
                             "kg_used": r.get("kg_used"),
                             "n_graph_relations": len(r.get("graph_sources") or []),
                             "elapsed_sec": round(time.time() - t0, 1)})
        except Exception as e:
            n_errors += 1
            out_rows.append({**row,
                             "answer": f"ERROR: {type(e).__name__}: {e}",
                             "sources": "", "kg_used": "", "n_graph_relations": "",
                             "elapsed_sec": round(time.time() - t0, 1)})

    fieldnames = list(rows[0].keys()) + [c for c in _RESULT_COLS if c not in rows[0]]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)
    return {"n": len(out_rows), "n_errors": n_errors, "csv": buf.getvalue()}

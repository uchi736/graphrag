"""EDC ナレッジグラフ抽出 HTTP API（FastAPI・自己完結）。

他のRAGアプリ / Dify 等から HTTP で呼べる薄いサービス。OpenAPI(/openapi.json, /docs)を
自動公開するので、Dify のカスタムツール(OpenAPIスキーマ取込)としてそのまま登録できる。
CWD非依存（テンプレ/スキーマを絶対パス解決）・副作用なし（ファイル出力なし）。
E→D→C本体・run.py・app.py は無改変。

起動:
    uvicorn api:app --host 0.0.0.0 --port 8080
    # /docs (Swagger UI), /openapi.json (Dify取込用)

エンドポイント:
    GET  /health        ヘルスチェック
    GET  /doctypes      登録済み文書タイプ一覧
    POST /classify      文書タイプ分類のみ
    POST /extract       テキスト→KG抽出（doctypeで標準スキーマ自動適用）
    POST /extract_file  ファイル(.txt/.md/.pdf)アップロード→KG抽出
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent  # プロジェクトルート（prompt_templates等がある場所）
_env = _ROOT / ".env"
if _env.exists():
    load_dotenv(_env)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _p(*parts) -> str:
    return str(_ROOT.joinpath(*parts))


# ============================================================
# パイプライン（CWD非依存・副作用なし）
# ============================================================
def _build_config(llm_model="vllm", embedder="vllm",
                  target_schema_path=None, target_types_path=None,
                  enrich_schema=False, enrich_types=False) -> dict:
    return {
        "oie_llm": llm_model,
        "oie_prompt_template_file_path": _p("prompt_templates", "oie_template.txt"),
        "oie_few_shot_example_file_path": _p("few_shot_examples", "example", "oie_few_shot_examples.txt"),
        "sd_llm": llm_model,
        "sd_prompt_template_file_path": _p("prompt_templates", "sd_template.txt"),
        "sd_few_shot_example_file_path": _p("few_shot_examples", "example", "sd_few_shot_examples.txt"),
        "sc_llm": llm_model,
        "sc_embedder": embedder,
        "sc_prompt_template_file_path": _p("prompt_templates", "sc_template.txt"),
        "sr_adapter_path": None,
        "sr_embedder": embedder,
        "oie_refine_prompt_template_file_path": _p("prompt_templates", "oie_r_template.txt"),
        "oie_refine_few_shot_example_file_path": _p("few_shot_examples", "example", "oie_few_shot_refine_examples.txt"),
        "ee_llm": llm_model,
        "ee_prompt_template_file_path": _p("prompt_templates", "ee_template.txt"),
        "ee_few_shot_example_file_path": _p("few_shot_examples", "example", "ee_few_shot_examples.txt"),
        "em_prompt_template_file_path": _p("prompt_templates", "em_template.txt"),
        "oie_typed_prompt_template_file_path": _p("prompt_templates", "oie_typed_template.txt"),
        "oie_typed_few_shot_example_file_path": _p("few_shot_examples", "example", "oie_typed_few_shot_examples.txt"),
        "tc_prompt_template_file_path": _p("prompt_templates", "tc_template.txt"),
        "target_schema_path": target_schema_path,
        "enrich_schema": enrich_schema,
        "target_types_path": target_types_path,
        "enrich_types": enrich_types,
        "loglevel": None,
    }


def _split_by_line(text):
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _split_by_heading(text):
    chunks, cur = [], []
    for line in text.split("\n"):
        if re.match(r'^#{1,6}\s', line) and cur:
            c = "\n".join(cur).strip()
            if c:
                chunks.append(c)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        c = "\n".join(cur).strip()
        if c:
            chunks.append(c)
    return chunks if chunks else _split_by_line(text)


def _split_recursive(text, chunk_size=1000, overlap=100):
    seps = ["\n\n", "\n", "。", ".", " "]

    def _s(t, i):
        if len(t) <= chunk_size:
            return [t.strip()] if t.strip() else []
        if i >= len(seps):
            return [t[j:j + chunk_size].strip() for j in range(0, len(t), chunk_size - overlap) if t[j:j + chunk_size].strip()]
        out, cur = [], ""
        for p in t.split(seps[i]):
            cand = cur + seps[i] + p if cur else p
            if len(cand) <= chunk_size:
                cur = cand
            else:
                if cur.strip():
                    out.append(cur.strip())
                if len(p) > chunk_size:
                    out.extend(_s(p, i + 1)); cur = ""
                else:
                    cur = p
        if cur.strip():
            out.append(cur.strip())
        return out
    r = _s(text, 0)
    return r if r else _split_by_line(text)


def _chunk(text, method="line", chunk_size=1000):
    if method == "heading":
        return _split_by_heading(text)
    if method == "recursive":
        return _split_recursive(text, chunk_size)
    return _split_by_line(text)


def _assemble_inputs(text=None, file_bytes=None, filename=None, chunk_method="line", chunk_size=1000):
    import pdf_processor
    is_pdf = bool(filename) and filename.lower().endswith(".pdf")
    if text is not None:
        return _chunk(text, chunk_method, chunk_size)
    if file_bytes is not None:
        if is_pdf:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes); tmp_path = tmp.name
            try:
                return pdf_processor.extract_text_from_pdf(tmp_path)  # ページ毎
            finally:
                os.unlink(tmp_path)
        return _chunk(file_bytes.decode("utf-8"), chunk_method, chunk_size)
    raise ValueError("text または file が必要です")


def _format_triplets(results):
    out = []
    for chunk in results:
        for t in chunk:
            if t is None:
                continue
            if len(t) == 3:
                out.append({"subject": t[0], "relation": t[1], "object": t[2]})
            elif len(t) == 5:
                out.append({"subject": t[0], "subject_type": t[1], "relation": t[2],
                            "object": t[3], "object_type": t[4]})
    return out


def extract_kg(text=None, file_bytes=None, filename=None, doctype=None,
               chunk_method="line", chunk_size=1000, llm_model="vllm", embedder="vllm",
               enrich_schema=None, enrich_types=None) -> dict:
    """文書からKG(トリプル)を抽出。doctypeで文書タイプ分類→標準スキーマ自動適用。"""
    input_texts = _assemble_inputs(text, file_bytes, filename, chunk_method, chunk_size)
    if not input_texts:
        return {"doctype": None, "classification": None, "schema": {}, "types": {},
                "n_input_chunks": 0, "n_triplets": 0, "triplets": []}

    dt_name = schema_path = types_path = cls = None
    if doctype:
        from doctype_router import load_registry, resolve
        reg = load_registry()
        sample = "\n".join(input_texts[:40])[:3000]
        dt_name, schema_path, types_path, cls = resolve(sample, reg, doctype=doctype)

    # 既定: 関係は常に enrich(=フリー発見でも空スキーマに全ドロップしない)。
    # 型は doctype で型スキーマが来たときだけ既定ON(=typed)。指定があれば尊重。
    if enrich_schema is None:
        enrich_schema = True
    if enrich_types is None:
        enrich_types = bool(types_path)
    enrich_schema, enrich_types = bool(enrich_schema), bool(enrich_types)

    from edc.edc_framework import EDC
    edc = EDC(**_build_config(llm_model, embedder, schema_path, types_path, enrich_schema, enrich_types))
    A = edc.extract_and_define(input_texts)
    results = edc.canonicalize_with_schema(
        input_texts, A["oie_triplets_list"], A["sd_dict_list"], dict(edc.schema),
        typed_oie_list=A["typed_oie_list"], enrich=enrich_schema,
        curated_types=(dict(edc.types) if edc.types else None), enrich_types=enrich_types,
    )
    triplets = _format_triplets(results)
    return {
        "doctype": dt_name, "classification": cls,
        "schema": dict(edc.schema), "types": dict(edc.types),
        "n_input_chunks": len(input_texts), "n_triplets": len(triplets), "triplets": triplets,
    }


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="EDC Knowledge Graph Extraction API",
    description="文書からトリプル(KG)を抽出。文書タイプ分類→標準スキーマ自動適用に対応（完全ローカル）。",
    version="1.0.0",
)


class ExtractRequest(BaseModel):
    text: str
    doctype: Optional[str] = "auto"          # None / "auto" / "<タイプ名>"
    chunk_method: str = "line"               # line / heading / recursive
    enrich_schema: Optional[bool] = None
    enrich_types: Optional[bool] = None


class Triplet(BaseModel):
    subject: str
    relation: str
    object: str
    subject_type: Optional[str] = None
    object_type: Optional[str] = None


class ExtractResponse(BaseModel):
    doctype: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None
    schema_: Dict[str, str] = {}
    types: Dict[str, str] = {}
    n_input_chunks: int = 0
    n_triplets: int = 0
    triplets: List[Triplet] = []


def _to_response(r: dict) -> dict:
    return {
        "doctype": r.get("doctype"),
        "classification": r.get("classification"),
        "schema_": r.get("schema", {}),
        "types": r.get("types", {}),
        "n_input_chunks": r.get("n_input_chunks", 0),
        "n_triplets": r.get("n_triplets", 0),
        "triplets": r.get("triplets", []),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/doctypes")
def doctypes():
    from doctype_router import load_registry
    reg = load_registry()
    return [{"name": d["name"], "description": d.get("description", "")} for d in reg.get("doctypes", [])]


@app.post("/classify")
def classify(req: ExtractRequest):
    from doctype_router import load_registry, classify_document
    return classify_document(req.text, load_registry())


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    return _to_response(extract_kg(
        text=req.text, doctype=req.doctype, chunk_method=req.chunk_method,
        enrich_schema=req.enrich_schema, enrich_types=req.enrich_types,
    ))


@app.post("/extract_file", response_model=ExtractResponse)
async def extract_file(file: UploadFile = File(...), doctype: str = Form("auto"),
                       chunk_method: str = Form("line")):
    data = await file.read()
    return _to_response(extract_kg(
        file_bytes=data, filename=file.filename, doctype=doctype, chunk_method=chunk_method,
    ))

"""文書タイプ分類ルーター（EDCの前段）。

文書サンプルをLLMで分類し、該当タイプの標準スキーマ（不変骨格＋タイプ別ドメイン層）へ
ルーティングする。スキーマ自体は事前に schemas/registry.json で固定（人手キュレーション済み）。
LLMの役割は「どのタイプか」の分類のみで、従来のEDCパイプライン本体は無改変。

使い方:
    reg = load_registry()
    res = classify_document(sample_text, reg)              # {type, confidence, reason}
    schema_path, types_path = compose_target(res["type"], reg)  # 骨格∪ドメインの一時CSVパス
"""

import os
import csv
import json
import tempfile

import edc.utils.llm_utils as llm_utils


def load_registry(path: str = None) -> dict:
    """registry.json を読み込む（既定は本ファイルと同階層の schemas/registry.json）。"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas", "registry.json")
    with open(path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    reg["_base_dir"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))  # registry のあるルート
    return reg


def classify_document(text_sample: str, registry: dict, model: str = "vllm") -> dict:
    """文書サンプルを registry の文書タイプへ分類。該当なしは UNKNOWN。"""
    doctypes = registry.get("doctypes", [])
    reg_str = "\n".join("- %s: %s" % (d["name"], d.get("description", "")) for d in doctypes)
    prompt = (
        "以下の文書タイプ一覧から、与えた文書サンプルが最も該当するタイプを1つ選べ。"
        "該当が無ければ UNKNOWN。\n\n"
        "# 文書タイプ一覧\n%s\n\n"
        "# 文書サンプル\n%s\n\n"
        '# 出力(JSONのみ、説明文なし): {"type":"<タイプ名 or UNKNOWN>","confidence":0-1,"reason":"根拠を一文"}'
        % (reg_str, text_sample[:3000])
    )
    raw = llm_utils.openai_chat_completion(model, None, [{"role": "user", "content": prompt}],
                                           temperature=0, max_tokens=300)
    return _parse_json_obj(raw)


def _parse_json_obj(raw: str) -> dict:
    """LLM出力から最初のJSONオブジェクトを頑健に抽出。失敗時は UNKNOWN。"""
    s = raw.strip()
    if "```" in s:  # ```json ... ``` フェンス除去
        s = s.split("```")[1] if s.split("```")[1].strip().startswith("{") else s
        s = s.replace("json", "", 1) if s.lstrip().startswith("json") else s
    try:
        l, r = s.index("{"), s.rindex("}")
        obj = json.loads(s[l:r + 1])
    except Exception:
        return {"type": "UNKNOWN", "confidence": 0.0, "reason": "parse failed: " + raw[:100]}
    obj.setdefault("type", "UNKNOWN")
    obj.setdefault("confidence", None)
    obj.setdefault("reason", "")
    return obj


def _read_kv_csv(path: str) -> dict:
    """relation/type, definition の2列CSVを dict で返す。"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def _find_doctype(name: str, registry: dict):
    for d in registry.get("doctypes", []):
        if d["name"] == name:
            return d
    return None


def compose_target(doctype_name: str, registry: dict):
    """不変骨格 ∪ タイプ別ドメイン を結合した一時CSVを書き出し、(schema_path, types_path) を返す。

    UNKNOWN / 未登録タイプは (None, None) を返す（呼び出し側はフリー発見にフォールバック）。
    """
    dt = _find_doctype(doctype_name, registry)
    if dt is None:
        return None, None
    base = registry.get("_base_dir", ".")

    def _merge(backbone_rel, domain_rel):
        bb = _read_kv_csv(os.path.join(base, registry["backbone"][backbone_rel]))
        dm = _read_kv_csv(os.path.join(base, dt[domain_rel]))
        merged = dict(bb)
        merged.update(dm)  # ドメイン側を優先（同名は上書き）
        fd, p = tempfile.mkstemp(suffix=".csv", prefix="edc_%s_" % backbone_rel)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for k, v in merged.items():
                w.writerow([k, v])
        return p

    schema_path = _merge("schema", "schema")
    types_path = _merge("types", "types")
    return schema_path, types_path


def resolve(text_sample: str, registry: dict, doctype: str = "auto", model: str = "vllm"):
    """高水準ヘルパ: doctype=='auto'なら分類、そうでなければ指定タイプを使い、
    (doctype_name, schema_path, types_path, classify_result) を返す。"""
    result = None
    if doctype == "auto":
        result = classify_document(text_sample, registry, model=model)
        name = result.get("type", "UNKNOWN")
    else:
        name = doctype
    schema_path, types_path = compose_target(name, registry)
    return name, schema_path, types_path, result

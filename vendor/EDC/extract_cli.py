"""EDC抽出のCLIラッパ（HTTPサーバ不要のワンショット実行）。

graphrag のスキーマ同期ジョブが子プロセスとして呼ぶ入口。api.py の
extract_kg()（/extract と同一実装）を1回実行して結果JSONをファイルへ書く。

    echo '{"text": "...", "doctype": "auto"}' | python extract_cli.py --out result.json

- 入力: stdin に JSON（/extract のリクエストと同形: text, doctype,
  chunk_method, enrich_schema, enrich_types）
- 出力: --out のファイルに JSON（/extract のレスポンスと同形。schema_ キー含む）
  ※ stdout はEDCパイプラインのログで汚れる可能性があるためファイル渡しにする
- 接続先(VLLM_*)は環境変数で注入する（.env が無ければ親プロセスの環境を継承）
"""
import argparse
import json
import sys
from pathlib import Path

from api import extract_kg, _to_response


def main() -> int:
    ap = argparse.ArgumentParser(description="EDC extract one-shot CLI")
    ap.add_argument("--out", required=True, help="結果JSONの出力先ファイル")
    args = ap.parse_args()

    req = json.load(sys.stdin)
    result = extract_kg(
        text=req.get("text"),
        doctype=req.get("doctype", "auto"),
        chunk_method=req.get("chunk_method", "line"),
        enrich_schema=req.get("enrich_schema"),
        enrich_types=req.get("enrich_types"),
    )
    Path(args.out).write_text(
        json.dumps(_to_response(result), ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""GraphRAG HTTP API（FastAPI 配送層）。

起動:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

注意: ジョブ管理がプロセス内のため **workers=1 で起動すること**。
ビジネスロジックは graphrag_core.services に置き、ここは HTTP 変換のみ。
"""

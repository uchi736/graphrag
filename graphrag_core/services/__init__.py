"""GraphRAG サービス層（UI フレームワーク非依存）。

Streamlit（graphrag_core.ui）と FastAPI（api/）の双方から呼ばれる
ビジネスロジックを置く。**このパッケージ配下で streamlit を import しないこと。**
進捗通知は services.progress.ProgressFn コールバックで抽象化する。
"""

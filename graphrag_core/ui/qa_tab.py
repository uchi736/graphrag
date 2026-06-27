"""
graphrag_core/ui/qa_tab.py
『💬 質問応答』タブ。render_qa_tab(ctx) + Langfuse トレースラッパー + 結果描画。
"""
import streamlit as st

from graphrag_core.ui.feedback import show_error
from graphrag_core.llm.langfuse_utils import observe, propagate_attributes
from graphrag_core.config import get_settings
from graphrag_core.text.japanese import SUDACHI_AVAILABLE
from graphrag_core.graph.provenance import graph_collection_status


# =====================================================================
# Langfuse トレース用ルートラッパー
# =====================================================================
@observe(name="graphrag_qa")
def _run_traced_question(chain, question, langfuse_session_id):
    """1質問=1トレースとして階層化"""
    _lf_sid = f"{langfuse_session_id}_q{hash(question) % 10000:04d}"
    with propagate_attributes(session_id=_lf_sid):
        return chain.invoke(question)


def _render_qa_result(qa: dict):
    """永続化した QA 結果(回答+エビデンス)を描画する。

    qa = {"question", "result", "config", "rerank_log"}。
    ボタンブロックの外から呼ぶことで rerun しても回答が消えない。
    """
    result = qa.get("result", {})
    cfg = qa.get("config", {})

    st.markdown("### 📝 回答")
    st.markdown(result.get("answer", ""))

    # この回答を生成した検索設定（再現性のため）
    mode_labels = {'hybrid': 'ハイブリッド', 'vector': 'ベクトル', 'keyword': 'キーワード'}
    doc_label = mode_labels.get(cfg.get("search_mode", "hybrid"), 'ベクトル')
    kg_on = "ON" if cfg.get("enable_knowledge_graph") else "OFF"
    rr_on = "ON" if cfg.get("enable_rerank") else "OFF"
    st.caption(f"🔎 {doc_label}検索 · KG {kg_on} · リランク {rr_on} · top_k={cfg.get('retrieval_top_k', 5)}")

    vector_sources = result.get("vector_sources", [])
    kg_chunks = result.get("kg_source_chunks", [])
    graph_paths = result.get("graph_paths", [])
    graph_sources = result.get("graph_sources", [])
    st.caption(f"根拠: 文書 {len(vector_sources)}件 ・ KGチャンク {len(kg_chunks)}件 ・ 推論パス {len(graph_paths)}本")

    # 参照ドキュメント（主たる根拠 → 既定で展開）
    with st.expander(f"📚 参照ドキュメント ({doc_label}検索) — {len(vector_sources)}件", expanded=True):
        if vector_sources:
            for i, doc in enumerate(vector_sources, 1):
                source = doc.metadata.get('source', '')
                st.markdown(f"**[D{i}]**" + (f" 出典: {source}" if source else ""))
                st.markdown(doc.page_content)
                if i < len(vector_sources):
                    st.divider()
        else:
            st.info("ドキュメント検索結果なし")

    # ナレッジグラフ（推論パス/トリプル）
    with st.expander(f"🕸️ ナレッジグラフ (Graph RAG) — パス{len(graph_paths)}本", expanded=False):
        if graph_paths:
            for i, p in enumerate(graph_paths, 1):
                st.markdown(f"**[推論パス{i}]** {p.get('path_text', '')}")
        elif graph_sources:
            for i, triple in enumerate(graph_sources, 1):
                st.markdown(f"**[トリプル{i}]** `{triple.get('start')}` -[{triple.get('type')}]→ `{triple.get('end')}`")
        else:
            st.info("グラフ検索結果なし")

    # KGソースチャンク
    with st.expander(f"📄 KGソースチャンク (Graph RAG) — {len(kg_chunks)}件", expanded=False):
        if kg_chunks:
            for i, doc in enumerate(kg_chunks, 1):
                source = doc.metadata.get('source', 'KG')
                st.markdown(f"**[KG{i}]** 出典: {source}")
                st.markdown(doc.page_content)
                if i < len(kg_chunks):
                    st.divider()
        else:
            st.info("KGからの追加チャンクなし")

    # デバッグ情報（リランクログ + 抽出エンティティ）を1つにまとめる
    with st.expander("🛠 デバッグ情報", expanded=False):
        rerank_log = qa.get("rerank_log", [])
        if rerank_log:
            st.markdown("**パスリランキングログ:**")
            for line in rerank_log:
                st.markdown(line)
            log_text = "\n".join(line.replace("`", "").replace("**", "") for line in rerank_log)
            st.download_button(
                "ログをダウンロード", data=log_text,
                file_name="path_rerank.log", mime="text/plain",
                key="dl_rerank_log",
            )
        extracted = result.get("extracted_entities", {})
        if extracted:
            st.markdown("**抽出されたエンティティ:**")
            llm_ents = extracted.get("llm_entities", [])
            if llm_ents:
                st.markdown("LLM抽出: " + ", ".join(llm_ents))
            vector_ents = extracted.get("vector_entities", [])
            if vector_ents:
                st.markdown("ベクトル/キーワード検索:")
                for eid, score in vector_ents[:10]:
                    st.write(f"- {eid} (score: {score:.3f})")
            merged_ents = extracted.get("merged_entities", [])
            if merged_ents:
                st.markdown("グラフ検索に使用: " + ", ".join(merged_ents[:15]))
        if not rerank_log and not extracted:
            st.caption("デバッグ情報なし")


# =====================================================================
# タブ1: 質問応答
# =====================================================================
def render_qa_tab(ctx):
    st.header("💬 質問応答")

    if ctx.initialized:
        s = get_settings()

        # グラフ↔コレクション整合性チェック（不整合/出自不明なら KG をスキップしてベクトル+BM25のみ）
        _gcs = graph_collection_status(ctx.graph, ctx.pg_collection)
        if _gcs["status"] != "match":
            _src = _gcs["graph_collection"] or "不明"
            _why = "不整合" if _gcs["status"] == "mismatch" else "出自不明"
            st.warning(
                f"⚠️ グラフ（出自: {_src}）が現在のコレクション「{ctx.pg_collection}」と{_why}のため、"
                "KG探索をスキップしてベクトル+BM25のみで回答します。"
                "一致させるには「🛠️ 構築/取り込み」で再構築するか、出自が既知なら同タブで記録してください。"
            )

        # 🔎 検索設定（旧⚙️設定タブから移設: 回答品質の主要レバーを質問の近くに置く）
        with st.expander("🔎 検索設定", expanded=False):
            retrieval_top_k = st.slider(
                "検索結果数 (Top-K)",
                min_value=1,
                max_value=20,
                value=s.retrieval_top_k,
                step=1,
                help="RAG検索で取得するチャンク数。多いほど文脈が豊富になりますが、処理時間が増加します。"
            )
            st.session_state.retrieval_top_k = retrieval_top_k

            # リランキング（cross-encoder, RERANKER_ENABLED が master switch）。
            # 最強レバーなので Top-K の直下に配置。
            enable_rerank = st.checkbox(
                "リランキング（cross-encoder）",
                value=s.enable_rerank,
                help="検索結果をcross-encoderで再ランキング（最強レバー +11.8pt）。"
                     "vLLMリランカ未設定時はLLMにフォールバック。"
            )
            st.session_state.enable_rerank = enable_rerank

            # 日本語ハイブリッド検索
            if SUDACHI_AVAILABLE:
                enable_jp_search = st.checkbox(
                    "日本語ハイブリッド検索",
                    value=s.enable_japanese_search,
                    help="ベクトル検索とキーワード検索を組み合わせます（精度向上）"
                )

                if enable_jp_search:
                    search_mode = st.radio(
                        "検索モード",
                        ["ハイブリッド (推奨)", "ベクトルのみ", "キーワードのみ"],
                        help="ハイブリッド: RRFでスコア統合 / ベクトル: 意味検索 / キーワード: 全文検索"
                    )

                    mode_map = {
                        "ハイブリッド (推奨)": "hybrid",
                        "ベクトルのみ": "vector",
                        "キーワードのみ": "keyword"
                    }
                    st.session_state.search_mode = mode_map[search_mode]
                    st.session_state.enable_japanese_search = True
                else:
                    st.session_state.search_mode = "vector"
                    st.session_state.enable_japanese_search = False
            else:
                st.warning("⚠️ sudachipy未インストール")
                st.caption("ベクトル検索のみ使用します")
                with st.expander("インストール方法"):
                    st.code("pip install sudachipy sudachidict_core")
                st.session_state.search_mode = "vector"
                st.session_state.enable_japanese_search = False

            # ナレッジグラフ利用（回答時のグラフ探索）。
            # ラベルは「生成（構築）」ではなく回答時の利用を表す
            enable_knowledge_graph = st.checkbox(
                "回答にナレッジグラフを使用",
                value=s.enable_knowledge_graph,
                help="グラフ探索（エンティティ・関係）を回答生成に利用します。オフ時はベクトル検索のみ。"
            )
            st.session_state.enable_knowledge_graph = enable_knowledge_graph

        question = st.text_area(
            "質問を入力してください:",
            height=90,
            key="question_input",
            placeholder="例: ガス軸受の空気膜はどのように形成されるか？",
        )

        if st.button("🔍 質問する", type="primary", disabled=st.session_state.busy):
            if not question or not question.strip():
                st.warning("質問を入力してください")
            else:
                st.session_state.busy = True
                try:
                    with st.spinner("回答生成中..."):
                        result = _run_traced_question(
                            ctx.chain, question,
                            st.session_state.langfuse_session_id
                        )
                    # 結果と生成時の設定を永続化（rerunしても消えない）
                    qa = {
                        "question": question,
                        "result": result,
                        "config": {
                            "search_mode": st.session_state.get("search_mode", "hybrid"),
                            "enable_knowledge_graph": st.session_state.get("enable_knowledge_graph", True),
                            "enable_rerank": st.session_state.get("enable_rerank", True),
                            "retrieval_top_k": st.session_state.get("retrieval_top_k", 5),
                        },
                        "rerank_log": st.session_state.get("path_rerank_log", []),
                    }
                    st.session_state.last_qa = qa
                    st.session_state.qa_history.insert(0, qa)
                    st.session_state.qa_history = st.session_state.qa_history[:20]
                    st.session_state.busy = False
                    st.rerun()
                except Exception as e:
                    st.session_state.busy = False
                    show_error("質問応答でエラーが発生しました", e)

        # 永続化した最新の回答を（ボタンブロックの外で）描画 → rerunで消えない
        if st.session_state.get("last_qa"):
            _render_qa_result(st.session_state.last_qa)

        # 履歴（最新を除く過去の質問）
        history = st.session_state.get("qa_history", [])
        if len(history) > 1:
            with st.expander(f"🕑 履歴 ({len(history) - 1}件)", expanded=False):
                if st.button("🗑️ 履歴をクリア"):
                    st.session_state.qa_history = history[:1]
                    st.rerun()
                for past in history[1:]:
                    q = past.get("question", "")
                    with st.expander(f"Q: {q[:40]}", expanded=False):
                        st.markdown(past.get("result", {}).get("answer", ""))
    else:
        st.info("「🛠️ 構築/取り込み」タブでドキュメントを構築するか、既存グラフを読み込んでください")

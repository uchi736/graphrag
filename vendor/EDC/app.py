"""
EDC (Extract, Define, Canonicalize) - Streamlit Web UI
"""
import streamlit as st
import os
import sys
import tempfile
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pdf_processor import extract_text_from_pdf


def extract_text_from_pdf_upload(uploaded_file) -> list:
    """
    アップロードされたPDFをページごとのテキストに変換（オンプレ既定）。

    PDF_PROCESSOR / PDF_BACKEND env に従い pdf_processor.extract_text_from_pdf へ委譲。
    Streamlitのアップロードファイルを一時 .pdf に書き出してパスで処理する。

    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト

    Returns:
        ページごとのテキストリスト
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


def build_edc_config(llm_model, embedder_model):
    """EDC のコンフィグを構築（テンプレパスは edc ディレクトリ基準の相対パス）。"""
    return {
        "oie_llm": llm_model,
        "oie_prompt_template_file_path": "./prompt_templates/oie_template.txt",
        "oie_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_examples.txt",
        "sd_llm": llm_model,
        "sd_prompt_template_file_path": "./prompt_templates/sd_template.txt",
        "sd_few_shot_example_file_path": "./few_shot_examples/example/sd_few_shot_examples.txt",
        "sc_llm": llm_model,
        "sc_embedder": embedder_model,
        "sc_prompt_template_file_path": "./prompt_templates/sc_template.txt",
        "sr_adapter_path": None,
        "sr_embedder": embedder_model,
        "oie_refine_prompt_template_file_path": "./prompt_templates/oie_r_template.txt",
        "oie_refine_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_refine_examples.txt",
        "ee_llm": llm_model,
        "ee_prompt_template_file_path": "./prompt_templates/ee_template.txt",
        "ee_few_shot_example_file_path": "./few_shot_examples/example/ee_few_shot_examples.txt",
        "em_prompt_template_file_path": "./prompt_templates/em_template.txt",
        "target_schema_path": None,
        "enrich_schema": False,
        "loglevel": None,
    }


def run_in_edc_dir(fn, *args, **kwargs):
    """相対テンプレパス解決のため edc ディレクトリに移動して fn を実行。"""
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent)
    try:
        return fn(*args, **kwargs)
    finally:
        os.chdir(original_dir)


st.set_page_config(
    page_title="EDC - 知識トリプル抽出",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 EDC: Extract, Define, Canonicalize")
st.markdown("LLMベースの知識トリプル抽出フレームワーク")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ 設定")

    # Check if Azure is configured in .env
    azure_configured = bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT") and
        os.environ.get("AZURE_OPENAI_API_KEY")
    )

    # API Provider selection（完全ローカル運用では vLLM が既定）
    api_provider = st.selectbox(
        "API Provider",
        ["vLLM (ローカル)", "Azure OpenAI", "OpenAI"],
        index=0,
        help="使用するLLM APIを選択（既定: ローカルvLLM）"
    )

    if api_provider == "vLLM (ローカル)":
        st.subheader("vLLM 設定")
        st.success("✅ .env の VLLM_* 設定を使用（完全ローカル）")
        st.caption(f"LLM: {os.environ.get('VLLM_ENDPOINT', '(未設定)')} / {os.environ.get('VLLM_MODEL', '(未設定)')}")
        st.caption(f"Embedding: {os.environ.get('VLLM_EMBEDDING_ENDPOINT', '(未設定)')} / {os.environ.get('VLLM_EMBEDDING_MODEL', '(未設定)')}")
        # "vllm" を使うと llm_utils が .env の VLLM_* 設定を参照
        llm_model = "vllm"
        embedder_model = "vllm"

    elif api_provider == "Azure OpenAI":
        st.subheader("Azure OpenAI 設定")

        # Show status from .env
        if azure_configured:
            st.success("✅ .envから設定を読み込みました")

        azure_endpoint = st.text_input(
            "Azure Endpoint",
            value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            type="default",
            help="例: https://your-resource.openai.azure.com/"
        )
        azure_api_key = st.text_input(
            "Azure API Key",
            value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            type="password"
        )
        azure_api_version = st.text_input(
            "API Version",
            value=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        chat_deployment = st.text_input(
            "Chat Deployment Name",
            value=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4"),
            help="チャット用のデプロイメント名"
        )
        embedding_deployment = st.text_input(
            "Embedding Deployment Name",
            value=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
            help="埋め込み用のデプロイメント名"
        )

        # Set environment variables
        if azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if azure_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
        if azure_api_version:
            os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version
        if chat_deployment:
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = chat_deployment
        if embedding_deployment:
            os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = embedding_deployment

        # Use "azure" to use .env settings
        llm_model = "azure"
        embedder_model = "azure"

    else:
        st.subheader("OpenAI 設定")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_KEY", ""),
            type="password"
        )
        if openai_key:
            os.environ["OPENAI_KEY"] = openai_key

        llm_model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="使用するOpenAIモデル"
        )

        # Embedder selection for OpenAI
        st.subheader("Embedder 設定")
        embedder_option = st.selectbox(
            "Sentence Embedder",
            [
                "all-MiniLM-L6-v2 (軽量・推奨)",
                "all-mpnet-base-v2 (中量)",
                "intfloat/e5-mistral-7b-instruct (重い)"
            ]
        )

        embedder_map = {
            "all-MiniLM-L6-v2 (軽量・推奨)": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (中量)": "all-mpnet-base-v2",
            "intfloat/e5-mistral-7b-instruct (重い)": "intfloat/e5-mistral-7b-instruct"
        }
        embedder_model = embedder_map[embedder_option]

    st.divider()

    # PDF処理設定（完全ローカル: オンプレ PaddleX が既定）
    st.subheader("📄 PDF処理設定")
    pdf_processor = os.environ.get("PDF_PROCESSOR", "onprem")

    if pdf_processor == "onprem":
        st.success(f"✅ オンプレ処理: {os.environ.get('PDF_BACKEND', 'paddleocr_remote')}")
        st.caption(f"PaddleX: {os.environ.get('PADDLEX_ENDPOINT', '(未設定)')}")
    else:
        st.info(f"PDF_PROCESSOR={pdf_processor}")

    with st.expander("Azure Document Intelligence（ロールバック用）"):
        di_endpoint = st.text_input(
            "DI Endpoint",
            value=os.environ.get("AZURE_DI_ENDPOINT", ""),
            help="例: https://your-resource.cognitiveservices.azure.com/"
        )
        di_api_key = st.text_input(
            "DI API Key",
            value=os.environ.get("AZURE_DI_API_KEY", ""),
            type="password"
        )
        di_model = st.selectbox(
            "DI Model",
            ["prebuilt-layout", "prebuilt-read", "prebuilt-document"],
            index=0,
            help="prebuilt-layout推奨（テーブル・図対応）"
        )

        if di_endpoint:
            os.environ["AZURE_DI_ENDPOINT"] = di_endpoint
        if di_api_key:
            os.environ["AZURE_DI_API_KEY"] = di_api_key
        if di_model:
            os.environ["AZURE_DI_MODEL"] = di_model

    st.divider()
    st.caption("スキーマ拡張(enrich)は②正規化ステップで選択します。")

# Main content area
st.subheader("📁 ファイルアップロード")

uploaded_files = st.file_uploader(
    "テキスト/PDFファイル（複数選択可）",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help="テキストファイル（1行1テキスト）またはPDFファイル（オンプレ PaddleX で処理）- 複数選択可"
)

# Schema options
st.subheader("ターゲットスキーマ（オプション）")
use_schema = st.checkbox("ターゲットスキーマを使用", value=False, help="スキーマなしで実行するとエッジを自動発見します")

uploaded_schema = None
if use_schema:
    uploaded_schema = st.file_uploader(
        "スキーマファイル（.csv）",
        type=["csv"],
        help="relation,definition形式のCSVファイル"
    )

# Entity-type options (typed mode) — 初期型を与えて漏れは追加
st.subheader("エンティティ型（オプション）")
typed_mode_ui = st.checkbox(
    "エンティティ型も抽出する（typedモード）",
    value=False,
    help="ONで5つ組(主語,型,関係,目的語,型)を抽出。初期型スキーマを与えると、漏れた型は抽出時に自動追加され、後で人手キュレーションできます。",
)
uploaded_types = None
if typed_mode_ui:
    uploaded_types = st.file_uploader(
        "初期型スキーマ（.csv: type,definition）",
        type=["csv"],
        help="初期の型セット。空でも可（自由発見）。抽出・正規化で漏れた型は追加されます。",
    )

# 文書タイプ・ルーティング（前段。EDC本体は無改変。選択時は標準スキーマを初期適用）
st.subheader("文書タイプ（前段ルーティング・オプション）")
try:
    from doctype_router import load_registry
    _registry = load_registry()
    _doctype_names = [d["name"] for d in _registry.get("doctypes", [])]
except Exception:
    _registry = None
    _doctype_names = []
doctype_mode = st.selectbox(
    "文書タイプ",
    ["（使わない）", "自動判定(LLM)"] + _doctype_names,
    index=0,
    help="自動判定: 抽出時にLLMが文書タイプを分類し、該当タイプの標準スキーマ(骨格+ドメイン層)を初期スキーマとして適用。タイプ直接指定も可。",
    disabled=(_registry is None),
)

# ============================================================
# HITL 2フェーズ・ウィザード
#   ① 抽出してスキーマ発見 (OIE + Schema Definition)
#   → 人が発見スキーマをレビュー/統合/修正
#   ② 確定スキーマで正規化 (Schema Canonicalization → 確定KG)
# ============================================================
st.divider()

if "stage" not in st.session_state:
    st.session_state.stage = "idle"


def _validate_provider():
    if api_provider == "vLLM (ローカル)":
        if not os.environ.get("VLLM_ENDPOINT"):
            st.error("VLLM_ENDPOINT を .env に設定してください"); st.stop()
    elif api_provider == "Azure OpenAI":
        if not os.environ.get("AZURE_OPENAI_ENDPOINT") or not os.environ.get("AZURE_OPENAI_API_KEY"):
            st.error("Azure OpenAIのEndpointとAPI Keyを設定してください"); st.stop()
    else:
        if not os.environ.get("OPENAI_KEY"):
            st.error("OpenAI API Keyを設定してください"); st.stop()


def _read_uploaded_files(files):
    """アップロードファイル群を input_texts と file_boundaries に変換。"""
    input_texts, file_boundaries = [], []
    for uploaded_file in files:
        start_idx = len(input_texts)
        if uploaded_file.name.endswith('.pdf'):
            # PDF処理（オンプレ既定: PaddleX。PDF_PROCESSOR/PDF_BACKEND で切替）
            try:
                texts = extract_text_from_pdf_upload(uploaded_file)
                input_texts.extend(texts)
                st.info(f"📄 {uploaded_file.name}: {len(texts)}ページを抽出")
            except Exception as e:
                st.error(f"PDF処理エラー ({uploaded_file.name}): {str(e)}")
                continue
        else:
            texts = uploaded_file.read().decode("utf-8").strip().split("\n")
            input_texts.extend(texts)
            st.info(f"📄 {uploaded_file.name}: {len(texts)}行を読み込み")
        file_boundaries.append((start_idx, len(input_texts), uploaded_file.name))
    return input_texts, file_boundaries


# ---- Phase A: 抽出してスキーマ発見 ----
if st.button("① 抽出してスキーマ発見", type="primary", use_container_width=True):
    _validate_provider()
    if not uploaded_files:
        st.error("ファイルをアップロードしてください"); st.stop()

    with st.spinner("ファイルを処理中..."):
        input_texts, file_boundaries = _read_uploaded_files(uploaded_files)
    if not input_texts:
        st.error("処理可能なテキストがありません"); st.stop()
    st.success(f"合計 {len(input_texts)} テキストを {len(uploaded_files)} ファイルから読み込みました")

    # 任意: アップロードされたターゲットスキーマを編集テーブルの初期値に混ぜる
    def _parse_csv_schema(file):
        out = {}
        for line in file.read().decode("utf-8").strip().split("\n"):
            if "," in line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    out[parts[0].strip().strip('"')] = parts[1].strip().strip('"')
        return out

    seed_schema = _parse_csv_schema(uploaded_schema) if (use_schema and uploaded_schema is not None) else {}
    seed_types = _parse_csv_schema(uploaded_types) if (typed_mode_ui and uploaded_types is not None) else {}

    # 文書タイプ・ルーティング: 選択時はLLM分類→該当タイプの標準スキーマを初期スキーマに採用
    routed_typed = False
    if _registry is not None and doctype_mode != "（使わない）":
        from doctype_router import resolve, _read_kv_csv
        sample = "\n".join(input_texts[:40])[:3000]
        _mode = "auto" if doctype_mode == "自動判定(LLM)" else doctype_mode
        with st.spinner("文書タイプを判定中..."):
            dt_name, sp, tp, cls = resolve(sample, _registry, doctype=_mode)
        if cls is not None:
            st.info(f"📑 文書タイプ判定: **{dt_name}** (conf={cls.get('confidence')}) — {cls.get('reason','')}")
        if sp:
            seed_schema = _read_kv_csv(sp)
            seed_types = _read_kv_csv(tp)
            routed_typed = True
            st.success(f"標準スキーマを初期適用: {dt_name}（関係{len(seed_schema)} / 型{len(seed_types)}）")
        else:
            st.warning(f"該当タイプなし({dt_name}) → フリー発見にフォールバック")

    try:
        from edc.edc_framework import EDC
        with st.spinner("① OIE + スキーマ定義を実行中...（LLM呼び出し）"):
            edc = EDC(**build_edc_config(llm_model, embedder_model))
            if typed_mode_ui or routed_typed:
                # typedモードを有効化（initial_types_path 無しでも enrich_types で起動）
                edc.enrich_types = True
                edc.types = dict(seed_types)  # 初期型スキーマでOIEを誘導
            phaseA = run_in_edc_dir(edc.extract_and_define, input_texts)

        # アップロード済みスキーマを発見スキーマにマージ（重複は発見側を優先）
        discovered = {d["relation"]: d for d in phaseA["discovered_schema"]}
        for rel, defn in seed_schema.items():
            if rel not in discovered:
                discovered[rel] = {"relation": rel, "definition": defn, "count": 0}
        phaseA["discovered_schema"] = sorted(discovered.values(), key=lambda x: -x["count"])

        st.session_state.edc = edc
        st.session_state.input_texts = input_texts
        st.session_state.file_boundaries = file_boundaries
        st.session_state.phaseA = phaseA
        st.session_state.final = None
        st.session_state.stage = "curate"
        st.session_state.pop("schema_editor", None)  # 編集テーブルをリセット
        st.session_state.pop("types_editor", None)
    except Exception as e:
        st.error(f"抽出中にエラーが発生しました: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ---- 人手キュレーション + Phase B ----
if st.session_state.stage in ("curate", "finalized"):
    phaseA = st.session_state.phaseA
    st.divider()
    st.subheader("✏️ スキーマ・キュレーション（人手レビュー）")
    st.caption("関係名のリネーム=統合 / 定義の修正 / 不要な行は『採用』を外すか削除。確定スキーマで正規化します。")

    schema_df = pd.DataFrame(
        [{"採用": True, "relation": d["relation"], "definition": d["definition"], "count": d["count"]}
         for d in phaseA["discovered_schema"]]
    )
    edited_df = st.data_editor(
        schema_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "採用": st.column_config.CheckboxColumn("採用", help="正規化先スキーマに含める", default=True),
            "relation": st.column_config.TextColumn("relation", help="同名にリネームすると統合される"),
            "definition": st.column_config.TextColumn("definition", width="large"),
            "count": st.column_config.NumberColumn("count (OIE出現)", disabled=True),
        },
        key="schema_editor",
    )

    # draft トリプルのプレビュー（関係別・読み取り専用、判断補助）
    with st.expander("🔍 抽出されたdraftトリプル（関係別プレビュー）"):
        by_rel = {}
        for chunk in phaseA["oie_triplets_list"]:
            for t in chunk:
                if len(t) >= 3:
                    by_rel.setdefault(t[1], []).append((t[0], t[2]))
        for rel in sorted(by_rel, key=lambda r: -len(by_rel[r])):
            ex_str = ", ".join(f"{s}→{o}" for s, o in by_rel[rel][:5])
            st.markdown(f"**{rel}** ({len(by_rel[rel])}): {ex_str}")

    enrich = st.checkbox(
        "キュレーション外の関係も追加（enrich）",
        value=False,
        help="ONだと確定スキーマに無い関係も新規追加。OFFだとマッピングできない関係は除外。",
    )

    # ---- エンティティ型キュレーション（typed時のみ。関係スキーマと対称）----
    edited_types_df = None
    enrich_types_ui = True
    if phaseA.get("discovered_types") is not None:
        st.divider()
        st.subheader("🏷️ エンティティ型キュレーション（人手レビュー）")
        st.caption("型名のリネーム=統合 / 定義の修正 / 不要な行は『採用』を外すか削除。初期型に無い型は『漏れ追加』で取り込めます。")
        types_df = pd.DataFrame(
            [{"採用": True, "type": d["type"], "definition": d["definition"], "count": d["count"]}
             for d in phaseA["discovered_types"]]
        )
        edited_types_df = st.data_editor(
            types_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "採用": st.column_config.CheckboxColumn("採用", help="正規化先の型に含める", default=True),
                "type": st.column_config.TextColumn("type", help="同名にリネームすると統合される"),
                "definition": st.column_config.TextColumn("definition", width="large", help="型の説明（埋め込み検索に使用）"),
                "count": st.column_config.NumberColumn("count (OIE出現)", disabled=True),
            },
            key="types_editor",
        )
        enrich_types_ui = st.checkbox(
            "初期型に無い型も追加（enrich, 漏れ追加）",
            value=True,
            help="ONだと確定型に無いraw型も新規追加（初期を与えて漏れは追加）。OFFだとマッピングできない型は 'Other'。",
        )

    if st.button("② 確定スキーマで正規化", type="primary", use_container_width=True):
        curated_schema = {}
        for _, row in edited_df.iterrows():
            rel = str(row.get("relation", "") or "").strip()
            if not rel or not bool(row.get("採用", False)):
                continue
            curated_schema[rel] = str(row.get("definition", "") or "").strip()

        if not curated_schema and not enrich:
            st.error("採用された関係がありません。1つ以上採用するか enrich をONにしてください。")
            st.stop()

        # 確定型スキーマを構築（typed時）
        curated_types = None
        if edited_types_df is not None:
            curated_types = {}
            for _, row in edited_types_df.iterrows():
                tp = str(row.get("type", "") or "").strip()
                if not tp or not bool(row.get("採用", False)):
                    continue
                curated_types[tp] = str(row.get("definition", "") or "").strip()

        try:
            edc = st.session_state.edc
            with st.spinner("② 確定スキーマで正規化中...（LLM呼び出し）"):
                results = run_in_edc_dir(
                    edc.canonicalize_with_schema,
                    st.session_state.input_texts,
                    phaseA["oie_triplets_list"],
                    phaseA["sd_dict_list"],
                    curated_schema,
                    typed_oie_list=phaseA["typed_oie_list"],
                    enrich=enrich,
                    curated_types=curated_types,
                    enrich_types=enrich_types_ui,
                )
            st.session_state.final = {
                "results": results,
                "schema": dict(edc.schema),
                "types": dict(edc.types) if edited_types_df is not None else None,
            }
            st.session_state.stage = "finalized"
        except Exception as e:
            st.error(f"正規化中にエラーが発生しました: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ---- 確定結果の表示 + エクスポート ----
if st.session_state.stage == "finalized" and st.session_state.get("final"):
    results = st.session_state.final["results"]
    final_schema = st.session_state.final["schema"]
    input_texts = st.session_state.input_texts
    file_boundaries = st.session_state.file_boundaries

    st.divider()
    st.success("✅ 正規化完了!")
    st.subheader("📊 確定トリプル")

    def _triplet_row(t):
        if t is None:
            return None
        if len(t) == 3:
            return {"Subject": t[0], "Relation": t[1], "Object": t[2]}
        if len(t) == 5:
            return {"Subject": t[0], "S-Type": t[1], "Relation": t[2], "Object": t[3], "O-Type": t[4]}
        return None

    for start_idx, end_idx, filename in file_boundaries:
        with st.expander(f"📄 {filename} ({end_idx - start_idx}テキスト)", expanded=True):
            for idx in range(start_idx, end_idx):
                st.markdown(f"**テキスト {idx - start_idx + 1}:** {input_texts[idx][:100]}...")
                rows = [r for r in (_triplet_row(t) for t in (results[idx] or [])) if r]
                if rows:
                    st.table(rows)
                else:
                    st.caption("正規化されたトリプルなし")
                st.divider()

    # 関係サマリ
    st.subheader("📈 関係サマリ")
    edge_summary = {}
    for triplets in results:
        for t in triplets or []:
            if t is None:
                continue
            if len(t) == 3:
                rel, subj, obj = t[1], t[0], t[2]
            elif len(t) == 5:
                rel, subj, obj = t[2], t[0], t[3]
            else:
                continue
            info = edge_summary.setdefault(rel, {"count": 0, "definition": final_schema.get(rel, ""), "examples": []})
            info["count"] += 1
            if len(info["examples"]) < 3:
                info["examples"].append((subj, obj))

    if edge_summary:
        edge_data = []
        for rel, info in sorted(edge_summary.items(), key=lambda x: -x[1]["count"]):
            definition = info["definition"]
            if len(definition) > 50:
                definition = definition[:50] + "..."
            edge_data.append({
                "リレーション": rel,
                "定義": definition,
                "出現回数": info["count"],
                "例": ", ".join(f"{s}→{o}" for s, o in info["examples"][:2]),
            })
        st.table(edge_data)
    else:
        st.info("確定トリプルはありません")

    # 型サマリ（typed時のみ。関係サマリと対称）
    if st.session_state.final.get("types") is not None:
        st.subheader("🏷️ エンティティ型サマリ")
        type_summary = {}
        for triplets in results:
            for t in triplets or []:
                if t is None or len(t) != 5:
                    continue
                for ent, tp in ((t[0], t[1]), (t[3], t[4])):
                    info = type_summary.setdefault(tp, {"count": 0, "examples": []})
                    info["count"] += 1
                    if ent not in info["examples"] and len(info["examples"]) < 4:
                        info["examples"].append(ent)
        if type_summary:
            st.table([
                {"型": tp, "出現回数": info["count"], "例": ", ".join(info["examples"])}
                for tp, info in sorted(type_summary.items(), key=lambda x: -x[1]["count"])
            ])

    # エクスポート
    st.subheader("📥 エクスポート")
    export_data = []
    for start_idx, end_idx, filename in file_boundaries:
        file_data = {"file": filename, "texts": []}
        for idx in range(start_idx, end_idx):
            file_data["texts"].append({
                "input_text": input_texts[idx],
                "triplets": [t for t in (results[idx] or []) if t is not None],
            })
        export_data.append(file_data)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="JSONとしてダウンロード",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name="triplets.json",
            mime="application/json",
        )
    with col2:
        csv_lines = ["relation,definition,count"]
        for rel, info in sorted(edge_summary.items(), key=lambda x: -x[1]["count"]):
            definition = info["definition"].replace('"', '""')
            csv_lines.append(f'"{rel}","{definition}",{info["count"]}')
        st.download_button(
            label="スキーマCSVとしてダウンロード",
            data="\n".join(csv_lines),
            file_name="discovered_schema.csv",
            mime="text/csv",
        )

# Footer
st.divider()
st.markdown("""
---
**EDC Framework** - [GitHub](https://github.com/clear-nus/edc) |
論文: [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction](https://arxiv.org/abs/2404.03868)
""")

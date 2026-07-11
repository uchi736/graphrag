from argparse import ArgumentParser
from edc.edc_framework import EDC
from pdf_processor import extract_text_from_pdf
import os
import logging
import json
import csv
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def split_by_line(text: str) -> list:
    """1行1テキストで分割（従来動作）"""
    return [line.strip() for line in text.split("\n") if line.strip()]


def split_by_heading(text: str) -> list:
    """Markdownの見出し（#）単位で分割"""
    chunks = []
    current_chunk = []

    for line in text.split("\n"):
        if re.match(r'^#{1,6}\s', line) and current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks if chunks else split_by_line(text)


def split_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """再帰的にテキストを分割（段落 → 文 → 文字数）"""
    separators = ["\n\n", "\n", "。", ".", " "]

    def _split(text: str, sep_idx: int) -> list:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        if sep_idx >= len(separators):
            # 最終手段: 文字数で強制分割
            result = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    result.append(chunk)
            return result

        sep = separators[sep_idx]
        parts = text.split(sep)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    result = _split(text, 0)
    return result if result else split_by_line(text)


if __name__ == "__main__":
    parser = ArgumentParser()
    # OIE module setting
    parser.add_argument(
        "--oie_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for open information extraction."
    )
    parser.add_argument(
        "--oie_prompt_template_file_path",
        default="./prompt_templates/oie_template.txt",
        help="Promp template used for open information extraction.",
    )
    parser.add_argument(
        "--oie_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
        help="Few shot examples used for open information extraction.",
    )

    # Schema Definition setting
    parser.add_argument(
        "--sd_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for schema definition."
    )
    parser.add_argument(
        "--sd_prompt_template_file_path",
        default="./prompt_templates/sd_template.txt",
        help="Prompt template used for schema definition.",
    )
    parser.add_argument(
        "--sd_few_shot_example_file_path",
        default="./few_shot_examples/example/sd_few_shot_examples.txt",
        help="Few shot examples used for schema definition.",
    )

    # Schema Canonicalization setting
    parser.add_argument(
        "--sc_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM used for schema canonicaliztion verification.",
    )
    parser.add_argument(
        "--sc_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--sc_prompt_template_file_path",
        default="./prompt_templates/sc_template.txt",
        help="Prompt template used for schema canonicalization verification.",
    )

    # Refinement setting
    parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
    parser.add_argument(
        "--sr_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--oie_refine_prompt_template_file_path",
        default="./prompt_templates/oie_r_template.txt",
        help="Prompt template used for refined open information extraction.",
    )
    parser.add_argument(
        "--oie_refine_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_refine_examples.txt",
        help="Few shot examples used for refined open information extraction.",
    )
    parser.add_argument(
        "--ee_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for entity extraction."
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="./prompt_templates/ee_template.txt",
        help="Prompt templated used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="./few_shot_examples/example/ee_few_shot_examples.txt",
        help="Few shot examples used for entity extraction.",
    )
    parser.add_argument(
        "--em_prompt_template_file_path",
        default="./prompt_templates/em_template.txt",
        help="Prompt template used for entity merging.",
    )

    # Input setting
    parser.add_argument(
        "--input_dir",
        default="./datasets",
        help="Input directory containing .txt, .pdf, and/or .md files to process.",
    )
    parser.add_argument(
        "--target_schema_path",
        default=None,
        help="File containing the target schema to align to (.csv). "
             "If not specified, runs in schema-free discovery mode.",
    )
    parser.add_argument(
        "--chunk_method",
        choices=["line", "heading", "recursive"],
        default="line",
        help="Text chunking method. 'line': split by newline (default). "
             "'heading': split by Markdown headings. "
             "'recursive': recursive split by paragraph/sentence/char size.",
    )
    parser.add_argument(
        "--chunk_size",
        default=1000, type=int,
        help="Max chunk size in characters (only for --chunk_method recursive).",
    )
    parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
    parser.add_argument(
        "--enrich_schema",
        action="store_true",
        help="Whether un-canonicalizable relations should be added to the schema.",
    )
    # Type Canonicalization settings (symmetric to --target_schema_path / --enrich_schema)
    parser.add_argument(
        "--target_types_path",
        default=None,
        help="File containing the target entity types to align to (.csv). "
             "If not specified (and --enrich_types is not set), runs in untyped mode (3-tuple).",
    )
    parser.add_argument(
        "--enrich_types",
        action="store_true",
        help="Whether un-canonicalizable entity types should be added to the type schema.",
    )
    parser.add_argument(
        "--tc_prompt_template_file_path",
        default="./prompt_templates/tc_template.txt",
        help="Prompt template used for type canonicalization verification.",
    )
    parser.add_argument(
        "--oie_typed_prompt_template_file_path",
        default="./prompt_templates/oie_typed_template.txt",
        help="Prompt template for typed OIE (5-tuple extraction).",
    )
    parser.add_argument(
        "--oie_typed_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_typed_few_shot_examples.txt",
        help="Few-shot examples for typed OIE.",
    )

    # Provider setting
    parser.add_argument(
        "--provider",
        choices=["azure", "openai", "local", "vllm"],
        default=None,
        help="LLM provider. 'azure' sets all LLM/embedder to use Azure OpenAI. "
             "'local' uses default HuggingFace models. "
             "'vllm' uses vLLM (OpenAI-compatible API server). "
             "未指定時は .env の LLM_PROVIDER を使用（完全ローカル運用では vllm）。",
    )

    # Document-type router (前段): 文書を分類し、該当タイプの標準スキーマを自動選択
    parser.add_argument(
        "--doctype",
        default=None,
        help="文書タイプ・ルーティング。'auto'で入力をLLM分類し schemas/registry.json の"
             "該当タイプの標準スキーマ(骨格+ドメイン層)を自動適用。タイプ名を直接指定も可。"
             "未指定なら従来通り(--target_schema_path/フリー発見)。",
    )

    # Output setting
    parser.add_argument("--output_dir", default="./output/tmp", help="Directory to output to.")
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    args = vars(args)

    # --provider 未指定時は .env の LLM_PROVIDER を使用（完全ローカル運用では vllm）
    if args["provider"] is None:
        args["provider"] = os.environ.get("LLM_PROVIDER")

    # --provider でLLM/Embedderを一括上書き
    if args["provider"] == "azure":
        for key in ["oie_llm", "sd_llm", "sc_llm", "ee_llm", "sc_embedder", "sr_embedder"]:
            args[key] = "azure"
    elif args["provider"] == "vllm":
        for key in ["oie_llm", "sd_llm", "sc_llm", "ee_llm", "sc_embedder", "sr_embedder"]:
            args[key] = "vllm"

    # --- Phase 1: 入力フォルダからファイル収集 + 読み込み ---
    input_dir = args["input_dir"]
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        exit(1)

    input_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir)
         if f.lower().endswith((".txt", ".pdf", ".md"))]
    )

    if not input_files:
        print(f"ERROR: No .txt, .pdf, or .md files found in: {input_dir}")
        exit(1)

    print(f"Found {len(input_files)} file(s) in '{input_dir}'")

    input_texts = []
    file_boundaries = []  # [(start_idx, end_idx, filename), ...]

    for file_path in input_files:
        start_idx = len(input_texts)
        filename = os.path.basename(file_path)

        if file_path.lower().endswith(".pdf"):
            try:
                texts = extract_text_from_pdf(file_path)
                input_texts.extend(texts)
                print(f"  PDF '{filename}': {len(texts)} pages extracted")
            except Exception as e:
                print(f"  ERROR processing PDF '{filename}': {e}")
                continue
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunk_method = args["chunk_method"]
            if chunk_method == "heading":
                texts = split_by_heading(content)
            elif chunk_method == "recursive":
                texts = split_recursive(content, chunk_size=args["chunk_size"])
            else:
                texts = split_by_line(content)
            input_texts.extend(texts)
            print(f"  Text '{filename}': {len(texts)} chunks ({chunk_method})")

        end_idx = len(input_texts)
        file_boundaries.append((start_idx, end_idx, filename))

    if not input_texts:
        print("ERROR: No input texts found. Exiting.")
        exit(1)

    print(f"Total: {len(input_texts)} texts from {len(file_boundaries)} file(s)")

    # --- Phase 1.5: 文書タイプ・ルーティング（前段。EDC本体は無改変）---
    if args.get("doctype"):
        from doctype_router import load_registry, resolve
        registry = load_registry()
        sample = "\n".join(input_texts[:40])[:3000]
        dt_name, schema_path, types_path, cls = resolve(sample, registry, doctype=args["doctype"])
        if cls is not None:
            print(f"[doctype] 分類: {dt_name} (conf={cls.get('confidence')}) - {cls.get('reason','')}")
        if schema_path:
            args["target_schema_path"] = schema_path
            args["enrich_schema"] = True  # ドメイン外は漏れ追加(類似度マージゲートで抑制)
            args["target_types_path"] = types_path
            args["enrich_types"] = True
            print(f"[doctype] 標準スキーマ適用: {dt_name} (schema/types を自動セット)")
        else:
            print(f"[doctype] 該当タイプなし({dt_name}) → 従来設定/フリー発見にフォールバック")

    # --- Phase 2: EDC実行 ---
    edc = EDC(**args)
    results = edc.extract_kg(
        input_texts,
        args["output_dir"],
        refinement_iterations=args["refinement_iterations"],
    )

    # --- Phase 3: triplets.json 出力 ---
    output_dir = args["output_dir"]

    export_data = []
    for start_idx, end_idx, filename in file_boundaries:
        file_data = {"file": filename, "texts": []}
        for idx in range(start_idx, end_idx):
            file_data["texts"].append({
                "input_text": input_texts[idx],
                "triplets": [t for t in results[idx] if t is not None]
            })
        export_data.append(file_data)

    triplets_path = os.path.join(output_dir, "triplets.json")
    with open(triplets_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"Triplets exported to: {triplets_path}")

    # --- Phase 4: discovered_schema.csv 出力 ---
    edge_summary = {}
    for triplets in results:
        for t in triplets:
            if t is None:
                continue
            if len(t) == 3:
                rel = t[1]
                subj, obj = t[0], t[2]
            elif len(t) == 5:
                rel = t[2]
                subj, obj = t[0], t[3]
            else:
                continue
            if rel not in edge_summary:
                edge_summary[rel] = {"count": 0, "definition": "", "examples": []}
            edge_summary[rel]["count"] += 1
            if len(edge_summary[rel]["examples"]) < 3:
                edge_summary[rel]["examples"].append((subj, obj))

    # Get definitions from EDC's (potentially enriched) schema
    for rel in edge_summary:
        if rel in edc.schema:
            edge_summary[rel]["definition"] = edc.schema[rel]

    schema_path = os.path.join(output_dir, "discovered_schema.csv")
    with open(schema_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["relation", "definition", "count"])
        for rel, info in sorted(edge_summary.items(), key=lambda x: -x[1]["count"]):
            writer.writerow([rel, info["definition"], info["count"]])
    print(f"Schema exported to: {schema_path}")

    # --- Phase 4b: discovered_types.csv (typed mode, symmetric to discovered_schema.csv) ---
    typed_mode = args.get("target_types_path") is not None or args.get("enrich_types", False)
    if typed_mode:
        type_summary = {}
        for triplets in results:
            for t in triplets:
                if t is None or len(t) != 5:
                    continue
                for type_name in (t[1], t[4]):
                    if type_name not in type_summary:
                        type_summary[type_name] = {"count": 0, "definition": ""}
                    type_summary[type_name]["count"] += 1

        for type_name in type_summary:
            if type_name in edc.types:
                type_summary[type_name]["definition"] = edc.types[type_name]

        types_path = os.path.join(output_dir, "discovered_types.csv")
        with open(types_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["entity_type", "definition", "count"])
            for type_name, info in sorted(type_summary.items(), key=lambda x: -x[1]["count"]):
                writer.writerow([type_name, info["definition"], info["count"]])
        print(f"Types exported to: {types_path}")

    # --- Phase 5: サマリー表示 ---
    print(f"\n=== Results Summary ===")
    print(f"Files processed: {len(file_boundaries)}")
    print(f"Total input texts: {len(input_texts)}")
    print(f"Unique relations discovered: {len(edge_summary)}")
    total_triplets = sum(info["count"] for info in edge_summary.values())
    print(f"Total triplets extracted: {total_triplets}")
    if typed_mode:
        print(f"Unique entity types discovered: {len(type_summary)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - iter*/result_at_each_stage.json (per iteration)")
    print(f"  - iter*/canon_kg.txt (per iteration)")
    print(f"  - triplets.json (clean export, grouped by file)")
    print(f"  - discovered_schema.csv (relation, definition, count)")
    if typed_mode:
        print(f"  - discovered_types.csv (entity_type, definition, count)")

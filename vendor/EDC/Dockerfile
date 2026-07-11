# EDC ナレッジグラフ抽出 API（FastAPI）。
# LLM/Embedding/OCR は外部(DGX Spark)を HTTP で呼ぶため、本イメージはスリム。
FROM python:3.12-slim

WORKDIR /app

# スリム依存のみ（vLLM経路。torch/transformers等は含めない）
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# アプリ本体 + リソース（必要なものだけ。datasets/output/docs/myenv等はコピーしない）
COPY api.py doctype_router.py pdf_processor.py ./
COPY edc ./edc
COPY prompt_templates ./prompt_templates
COPY few_shot_examples ./few_shot_examples
COPY schemas ./schemas

EXPOSE 8080

# .env は焼き込まず、実行時に環境変数(compose の env_file 等)で注入する
ENV TOKENIZERS_PARALLELISM=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

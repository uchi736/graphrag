# 画像生成・編集アプリ（DGX Spark / ComfyUI バックエンド）

DGX Spark 上の [ComfyUI](https://github.com/comfyanonymous/ComfyUI) を推論エンジンに、
オンプレミス完結で画像の **生成（text-to-image）** と **編集（instruction editing / img2img）**
を行う FastAPI アプリ。[実装計画書](../docs/) の第2〜3フェーズ（バックエンド + フロントエンド）を実装。

```
[ブラウザ UI (imagegen/frontend)]
      │ HTTP(JSON / multipart)
      ▼
[FastAPI (imagegen/backend)]  ← 本アプリの中核
      │ ・リクエスト検証／パラメータ整形
      │ ・ワークフロー JSON へのパラメータ差し込み
      │ ・ジョブキュー（単一ワーカー直列実行）／結果保存
      │ ComfyUI API (workflow JSON)
      ▼
[ComfyUI]  ← 推論（別プロセス。既定 127.0.0.1:8188）
```

## ディレクトリ構成

```
imagegen/
├── backend/
│   ├── main.py            FastAPI エントリポイント（+ フロント静的配信）
│   ├── config.py          環境変数からの設定
│   ├── schemas.py         リクエスト/レスポンスの Pydantic モデル
│   ├── comfy_client.py    ComfyUI HTTP API クライアント（投入・ポーリング・取得）
│   ├── workflows.py       ワークフロー差し込み（API フィールド ⇄ ノード対応表）
│   ├── workflows/
│   │   ├── generate.json  txt2img ワークフロー（ComfyUI API 形式）
│   │   └── edit.json      img2img 編集ワークフロー
│   ├── jobs.py            プロセス内ジョブキュー（queued→running→done/failed）
│   ├── pipeline.py        ジョブ実行本体（構築→投入→保存）
│   ├── routes.py          エンドポイント定義
│   └── state.py           アプリ共有状態
├── frontend/              React + Vite + TypeScript + Tailwind（生成/編集タブ・ポーリング）
│   ├── src/
│   │   ├── App.tsx            ヘッダ（ComfyUI 疎通表示）+ タブ切替
│   │   ├── api.ts / types.ts  型付き API クライアント
│   │   ├── hooks/             ジョブポーリング・ヘルス・モデル一覧
│   │   └── components/        GenerateTab / EditTab / JobResult / Field
│   └── dist/                  ビルド成果物（backend が配信。git 管理外）
├── tests/                 単体・スモークテスト（ComfyUI 不要）
├── requirements.txt
└── .env.sample
```

## 事前準備（計画書 第0〜1フェーズ：実機作業）

このリポジトリのコードは **バックエンドと UI**。推論には別途 ComfyUI が必要。

1. DGX Spark 上に ComfyUI を構築（最適化コンテナ／有志インストーラ
   [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) を土台に推奨）。
   - `--gpu-only` を付けない／グローバルな `--fp16-vae` 強制を避ける／SageAttention は 2.2 系。
2. 生成用・編集用モデル（例: Z-Image Turbo, Qwen-Image-Edit-2511, FLUX.1 Kontext, SDXL）を配置。
3. ComfyUI を API モードで起動（既定ポート 8188）。
4. `backend/workflows/*.json` を、実際に ComfyUI で組んだワークフローの
   **API 形式エクスポート**（「Save (API Format)」）に差し替える。
   差し替えたら `backend/workflows.py` の `mapping`（API フィールド → `(node_id, input_key)`）を
   新しいノード ID に合わせて更新する。

> 同梱の `generate.json` / `edit.json` は SDXL 相当の標準的な txt2img / img2img の
> ひな形。まずこれで疎通確認し、好みのモデルのワークフローに寄せていく。

## セットアップ & 起動

```bash
# 1) バックエンド依存
cd imagegen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env   # COMFYUI_URL 等を必要に応じて編集

# 2) フロントエンドをビルド（dist/ を backend が配信）
cd frontend
npm install
npm run build
cd ..

# 3) 起動（ジョブキューがプロセス内のため --workers 1 必須）
cd ..   # リポジトリルート（imagegen パッケージ解決のため）
python -m uvicorn imagegen.backend.main:app --host 0.0.0.0 --port 8100
```

ブラウザで <http://localhost:8100/> を開く。上部に ComfyUI 接続状態が出る。

### フロントエンド開発（ホットリロード）

```bash
cd imagegen/frontend
npm run dev        # Vite dev server (5173)。/generate 等は 8100 のバックエンドへプロキシ
# 別ターミナルで backend を 8100 で起動しておく（IMAGEGEN_BACKEND で接続先変更可）
```

## API（計画書 §8）

| メソッド | パス | 概要 | 主なパラメータ |
|----------|------|------|----------------|
| POST | `/generate` | 生成ジョブ投入（JSON） | `prompt, negative_prompt, width, height, steps, cfg, seed, model` |
| POST | `/edit` | 編集ジョブ投入（multipart） | `base_image`(file)`, prompt, negative_prompt, strength, steps, cfg, seed, model` |
| GET | `/jobs/{job_id}` | ジョブ状態取得 | — |
| GET | `/jobs` | ジョブ一覧 | — |
| POST | `/jobs/{job_id}/cancel` | ジョブキャンセル | — |
| GET | `/models` | 利用可能な checkpoint 一覧 | — |
| GET | `/images/{name}` | 結果画像の配信 | — |
| GET | `/health` | ComfyUI 疎通含むヘルス | — |

- `/generate`・`/edit` は即座に `202 {job_id, state}` を返す（非同期）。
- `/jobs/{job_id}` は `queued` / `running` / `done` / `failed` / `cancelled` と、
  完了時は `image_urls`（`/images/...`）を返す。
- `seed` は未指定／負値でランダム採番し、採番後の値を `params.seed` に載せて返す（再現用）。

### 例

```bash
# 生成
curl -s -X POST localhost:8100/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a serene mountain lake at dawn","steps":20}'
# -> {"job_id":"...","state":"queued"}

curl -s localhost:8100/jobs/<job_id>   # 状態ポーリング

# 編集
curl -s -X POST localhost:8100/edit \
  -F base_image=@input.png \
  -F prompt="make it winter with snow" \
  -F strength=0.6
```

## テスト

ComfyUI を差し替えてオフラインで検証できる。

```bash
# リポジトリルートから
pip install pytest
python -m pytest imagegen/tests -q
```

## パラメータ ⇄ ワークフローノード対応（既定テンプレート）

`backend/workflows.py` の `mapping` で定義。ワークフローを差し替えたらここを更新する。

| API フィールド | generate.json | edit.json |
|----------------|---------------|-----------|
| `prompt` | `6.text` | `6.text` |
| `negative_prompt` | `7.text` | `7.text` |
| `width` / `height` | `5.width` / `5.height` | —（ベース画像サイズに従う） |
| `steps` | `3.steps` | `3.steps` |
| `cfg` | `3.cfg` | `3.cfg` |
| `seed` | `3.seed` | `3.seed` |
| `strength` | — | `3.denoise` |
| `model` | `4.ckpt_name` | `4.ckpt_name` |
| `base_image`(upload) | — | `10.image` |

## 補足

- ジョブキューはプロセス内・単一ワーカーの直列実行（`--workers 1` 前提）。同時実行は当面 1 件。
- 結果画像は `outputs/`、アップロードは `uploads/`（`.gitignore` 済み。`IMAGEGEN_OUTPUT_DIR` 等で変更可）。
- モデル常駐やメモリ配分（計画書 第4フェーズ）は ComfyUI 側の設定で対応する。

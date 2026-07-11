# Vendored: EDC (Extract-Define-Canonicalize)

graphrag のスキーマ自動発見連携（`docs/EDC_INTEGRATION.md`）が使う EDC フレームワーク一式。
`git clone` 1回で連携に必要なコードが全部そろうよう、上流リポジトリの追跡ファイルを
そのままコピーしている（`git archive HEAD` 出力。venv・生成物・機密は含まない）。

- 上流: https://github.com/uchi736/EDC
- 取り込みコミット: `ecae3e16375eeb0a71544eb92e102e355db45066` (2026-07-08)
- 論文: Extract-Define-Canonicalize (arXiv:2404.03868)

## 起動（graphrag からの利用に必要なのは API だけ）

```bash
cd vendor/EDC
python -m venv .venv && .venv/Scripts/pip install -r requirements-api.txt   # 初回のみ
cp .env.example .env                                                         # 接続先を環境に合わせ編集
.venv/Scripts/python -m uvicorn api:app --host 127.0.0.1 --port 8080
```

graphrag 側は `EDC_ENDPOINT`（既定 `http://127.0.0.1:8080`）で接続する。

## 更新手順

上流で開発 → graphrag へ再取り込み:

```bash
git -C <EDCリポジトリ> archive HEAD | tar -x -C <graphrag>/vendor/EDC
# このファイルの「取り込みコミット」を更新してコミット
```

EDC 自体の開発・履歴は上流リポジトリで行うこと（ここは読み取り専用のスナップショット）。

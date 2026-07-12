"""ComfyUI API を叩く薄いクライアント。

計画書 §7 第2フェーズ「ComfyUI の API を叩く薄いクライアント（ワークフロー投入・
完了ポーリング）」。WebSocket は使わず HTTP ポーリングで完了を待つ。

参照する ComfyUI HTTP エンドポイント:
  POST /prompt                 ワークフロー投入 -> {prompt_id, ...}
  GET  /history/{prompt_id}    実行結果（outputs, status）
  GET  /view?filename&subfolder&type   画像バイト取得
  POST /upload/image           ベース画像アップロード -> {name, subfolder, type}
  GET  /object_info/{class}    ノード定義（利用可能な ckpt 一覧の取得に使う）
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import requests


class ComfyError(RuntimeError):
    pass


class ComfyTimeout(ComfyError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, http_timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.http_timeout = http_timeout
        # ComfyUI は client_id 単位でキュー／履歴を紐づける
        self.client_id = uuid.uuid4().hex

    # ---- 低レベル --------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ping(self) -> bool:
        """ComfyUI が生きているか（/system_stats で確認）。"""
        try:
            r = requests.get(self._url("/system_stats"), timeout=self.http_timeout)
            return r.ok
        except requests.RequestException:
            return False

    # ---- 投入・ポーリング ------------------------------------------------

    def submit(self, workflow: Dict[str, Any]) -> str:
        """ワークフローをキュー投入し prompt_id を返す。"""
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            r = requests.post(self._url("/prompt"), json=payload, timeout=self.http_timeout)
        except requests.RequestException as e:
            raise ComfyError(f"ComfyUI への接続失敗: {e}") from e
        if r.status_code != 200:
            # ComfyUI はバリデーション失敗を 400 + {error, node_errors} で返す
            detail = _safe_json(r)
            raise ComfyError(f"ワークフロー投入失敗 (HTTP {r.status_code}): {detail}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"prompt_id が返らなかった: {data}")
        return prompt_id

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """指定 prompt_id の history エントリ。未完了なら None。"""
        try:
            r = requests.get(self._url(f"/history/{prompt_id}"), timeout=self.http_timeout)
        except requests.RequestException as e:
            raise ComfyError(f"history 取得失敗: {e}") from e
        if not r.ok:
            raise ComfyError(f"history 取得失敗 (HTTP {r.status_code})")
        data = r.json()
        return data.get(prompt_id)

    def wait(
        self,
        prompt_id: str,
        poll_interval: float,
        timeout: float,
        should_cancel: Callable[[], bool] = lambda: False,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, Any]:
        """history に結果が現れるまでポーリングして entry を返す。

        Raises:
            ComfyTimeout: timeout 超過
            ComfyError:   ComfyUI 側でエラー終了 / キャンセル
        """
        deadline = now() + timeout
        while True:
            if should_cancel():
                raise ComfyError("ジョブがキャンセルされました")
            entry = self.get_history(prompt_id)
            if entry is not None:
                status = entry.get("status", {}) or {}
                status_str = status.get("status_str")
                if status_str == "error" or status.get("completed") is False:
                    raise ComfyError(f"ComfyUI 実行エラー: {status}")
                # 出力が揃っていれば完了
                if entry.get("outputs"):
                    return entry
                if status.get("completed") is True:
                    return entry
            if now() >= deadline:
                raise ComfyTimeout(f"タイムアウト（{timeout:.0f}s）: prompt_id={prompt_id}")
            sleep(poll_interval)

    # ---- 出力の取り出し --------------------------------------------------

    @staticmethod
    def extract_images(entry: Dict[str, Any], save_node: Optional[str] = None) -> List[Dict[str, str]]:
        """history entry から画像参照 [{filename, subfolder, type}] を集める。"""
        images: List[Dict[str, str]] = []
        outputs = entry.get("outputs", {}) or {}
        node_ids = [save_node] if save_node and save_node in outputs else list(outputs.keys())
        for nid in node_ids:
            for img in (outputs.get(nid, {}) or {}).get("images", []) or []:
                if img.get("type") == "temp":
                    continue  # プレビュー等は除外
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
        return images

    def download_image(self, ref: Dict[str, str]) -> bytes:
        """/view で画像バイトを取得する。"""
        params = {
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        }
        try:
            r = requests.get(self._url("/view"), params=params, timeout=self.http_timeout)
        except requests.RequestException as e:
            raise ComfyError(f"画像取得失敗: {e}") from e
        if not r.ok:
            raise ComfyError(f"画像取得失敗 (HTTP {r.status_code})")
        return r.content

    # ---- アップロード ----------------------------------------------------

    def upload_image(self, data: bytes, filename: str, overwrite: bool = True) -> str:
        """ベース画像を ComfyUI にアップロードし、LoadImage が参照する名前を返す。"""
        files = {"image": (filename, data)}
        form = {"overwrite": "true" if overwrite else "false"}
        try:
            r = requests.post(
                self._url("/upload/image"), files=files, data=form, timeout=self.http_timeout
            )
        except requests.RequestException as e:
            raise ComfyError(f"画像アップロード失敗: {e}") from e
        if not r.ok:
            raise ComfyError(f"画像アップロード失敗 (HTTP {r.status_code}): {_safe_json(r)}")
        info = r.json()
        name = info.get("name", filename)
        subfolder = info.get("subfolder", "")
        # LoadImage は "subfolder/name" 形式を受け付ける
        return f"{subfolder}/{name}" if subfolder else name

    # ---- モデル一覧 ------------------------------------------------------

    def list_checkpoints(self) -> List[str]:
        """CheckpointLoaderSimple の ckpt_name 候補一覧を返す。"""
        try:
            r = requests.get(
                self._url("/object_info/CheckpointLoaderSimple"), timeout=self.http_timeout
            )
        except requests.RequestException as e:
            raise ComfyError(f"モデル一覧取得失敗: {e}") from e
        if not r.ok:
            raise ComfyError(f"モデル一覧取得失敗 (HTTP {r.status_code})")
        data = r.json()
        try:
            options = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        except (KeyError, IndexError, TypeError):
            return []
        return list(options) if isinstance(options, (list, tuple)) else []


def _safe_json(r: "requests.Response") -> Any:
    try:
        return r.json()
    except ValueError:
        return r.text[:500]

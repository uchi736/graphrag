"""ジョブ実行パイプライン（ComfyUI 呼び出し + 結果保存）。

router から呼ばれ、JobQueue に渡す runner を組み立てる。runner はワーカースレッドで
実行され、ワークフロー構築 -> 投入 -> ポーリング -> 画像保存 を行う。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from .comfy_client import ComfyClient
from .config import Settings
from .jobs import ImageJob
from .workflows import build_workflow

logger = logging.getLogger(__name__)


def make_runner(
    kind: str,
    comfy: ComfyClient,
    settings: Settings,
) -> Callable[[ImageJob, Callable[[], bool]], List[Path]]:
    """kind（generate/edit）に対応する runner を返す。"""

    def runner(job: ImageJob, should_cancel: Callable[[], bool]) -> List[Path]:
        params = dict(job.params)

        # 編集: ベース画像を ComfyUI にアップロードし、LoadImage 参照名を差し込む
        base_bytes = params.pop("_base_image_bytes", None)
        if base_bytes is not None:
            filename = params.pop("_base_image_name", f"{job.id}.png")
            ref_name = comfy.upload_image(base_bytes, filename)
            params["base_image"] = ref_name

        workflow, resolved = build_workflow(kind, params)
        # 採番後 seed 等を記録（再現用）
        job.params.update({k: v for k, v in resolved.items() if not k.startswith("_")})

        prompt_id = comfy.submit(workflow)
        job.prompt_id = prompt_id
        logger.info("job %s submitted to ComfyUI: prompt_id=%s", job.id, prompt_id)

        entry = comfy.wait(
            prompt_id,
            poll_interval=settings.poll_interval,
            timeout=settings.job_timeout,
            should_cancel=should_cancel,
        )

        refs = comfy.extract_images(entry, save_node=_save_node(kind))
        if not refs:
            raise RuntimeError("ComfyUI から出力画像が得られなかった")

        settings.ensure_dirs()
        saved: List[Path] = []
        for i, ref in enumerate(refs):
            data = comfy.download_image(ref)
            suffix = Path(ref["filename"]).suffix or ".png"
            out_name = f"{job.id}_{i}{suffix}"
            out_path = settings.output_dir / out_name
            out_path.write_bytes(data)
            saved.append(out_path)
        logger.info("job %s saved %d image(s)", job.id, len(saved))
        return saved

    return runner


def _save_node(kind: str) -> str:
    from .workflows import WORKFLOWS
    return WORKFLOWS[kind].save_node


def build_params_from_request(req: Any) -> Dict[str, Any]:
    """Pydantic リクエスト -> パイプライン用 params dict。"""
    return req.model_dump()

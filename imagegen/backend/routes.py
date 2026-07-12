"""API ルート定義（計画書 §8）。

  POST /generate        生成ジョブ投入 -> job_id
  POST /edit            編集ジョブ投入（multipart: base_image + パラメータ）-> job_id
  GET  /jobs/{job_id}   ジョブ状態取得
  GET  /jobs            ジョブ一覧
  GET  /models          利用可能モデル一覧
  GET  /images/{name}   結果画像の配信
  GET  /health          ComfyUI 疎通を含むヘルス
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .comfy_client import ComfyError
from .jobs import ImageJob
from .pipeline import make_runner
from .schemas import (
    EditRequest,
    GenerateRequest,
    JobAccepted,
    JobStatus,
    ModelList,
)
from .state import AppState

router = APIRouter()


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx


def _job_status(job: ImageJob) -> JobStatus:
    snap = job.snapshot()
    urls = [f"/images/{Path(p).name}" for p in snap["images"]]
    return JobStatus(image_urls=urls, **snap)


# ---- 生成 / 編集 ---------------------------------------------------------


@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate(req: GenerateRequest, request: Request) -> JobAccepted:
    ctx = _ctx(request)
    runner = make_runner("generate", ctx.comfy, ctx.settings)
    job = ctx.queue.submit("generate", req.model_dump(), runner)
    return JobAccepted(job_id=job.id, state=job.state)


@router.post("/edit", response_model=JobAccepted, status_code=202)
async def edit(
    request: Request,
    base_image: UploadFile = File(..., description="ベース画像"),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.75),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
) -> JobAccepted:
    ctx = _ctx(request)
    # multipart フィールドを Pydantic で検証（範囲チェック等を共通化）
    req = EditRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        steps=steps,
        cfg=cfg,
        seed=seed,
        model=model,
    )
    data = await base_image.read()
    if not data:
        raise HTTPException(status_code=400, detail="base_image が空です")

    params = req.model_dump()
    # runner 内で ComfyUI にアップロードするため一時的に埋め込む（_ 接頭辞は非公開）
    params["_base_image_bytes"] = data
    params["_base_image_name"] = base_image.filename or "base.png"

    runner = make_runner("edit", ctx.comfy, ctx.settings)
    job = ctx.queue.submit("edit", params, runner)
    return JobAccepted(job_id=job.id, state=job.state)


# ---- 状態 / 一覧 ---------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str, request: Request) -> JobStatus:
    job = _ctx(request).queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return _job_status(job)


@router.get("/jobs")
def job_list(request: Request):
    return _ctx(request).queue.list()


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str, request: Request):
    ok = _ctx(request).queue.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="キャンセルできない状態です")
    return {"job_id": job_id, "cancelled": True}


# ---- モデル一覧 ----------------------------------------------------------


@router.get("/models", response_model=ModelList)
def models(request: Request) -> ModelList:
    ctx = _ctx(request)
    try:
        ckpts = ctx.comfy.list_checkpoints()
        return ModelList(checkpoints=ckpts)
    except ComfyError as e:
        # ComfyUI 未起動でも UI を壊さない
        return ModelList(checkpoints=[], note=f"ComfyUI に接続できません: {e}")


# ---- 画像配信 ------------------------------------------------------------


@router.get("/images/{name}")
def get_image(name: str, request: Request):
    ctx = _ctx(request)
    # パストラバーサル防止（ファイル名のみ許可）
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="不正なファイル名")
    path = ctx.settings.output_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path)


# ---- ヘルス --------------------------------------------------------------


@router.get("/health")
def health(request: Request):
    ctx = _ctx(request)
    comfy_ok = ctx.comfy.ping()
    return {
        "ok": True,
        "comfyui_url": ctx.settings.comfyui_url,
        "comfyui_reachable": comfy_ok,
    }

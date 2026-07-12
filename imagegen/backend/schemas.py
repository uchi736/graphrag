"""API リクエスト／レスポンスの Pydantic モデル。

フィールド名は計画書 §8 に合わせ、ワークフロー側のノード入力に 1:1 で差し込む。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """F-01 生成（ベースなし）。"""

    prompt: str = Field(..., min_length=1, description="生成プロンプト")
    negative_prompt: str = Field("", description="ネガティブプロンプト")
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)
    steps: int = Field(20, ge=1, le=150)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    # None または負値でランダムシード
    seed: Optional[int] = Field(None, description="None/負値でランダム")
    # None なら ComfyUI ワークフローの既定チェックポイントを使う
    model: Optional[str] = Field(None, description="ckpt ファイル名。未指定でテンプレート既定")


class EditRequest(BaseModel):
    """F-02 編集（ベースあり）。base_image はアップロード API 経由で受け取る。"""

    prompt: str = Field(..., min_length=1, description="編集指示プロンプト")
    negative_prompt: str = Field("", description="ネガティブプロンプト")
    # img2img の denoise。1.0 に近いほど大きく書き換わる
    strength: float = Field(0.75, ge=0.0, le=1.0)
    steps: int = Field(20, ge=1, le=150)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    seed: Optional[int] = Field(None, description="None/負値でランダム")
    model: Optional[str] = Field(None, description="ckpt ファイル名。未指定でテンプレート既定")


class JobAccepted(BaseModel):
    job_id: str
    state: str


class JobStatus(BaseModel):
    job_id: str
    kind: str                       # generate | edit
    state: str                      # queued | running | done | failed
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    prompt_id: Optional[str] = None  # ComfyUI 側の prompt_id
    error: Optional[str] = None
    # 完了時: 結果画像のパス（複数枚対応）とブラウザ表示用 URL
    images: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


class ModelList(BaseModel):
    checkpoints: list[str] = Field(default_factory=list)
    # ComfyUI 未接続時などに理由を載せる
    note: Optional[str] = None

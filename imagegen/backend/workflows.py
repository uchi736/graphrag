"""ワークフローJSONテンプレートの読込とパラメータ差し込み。

計画書 §8 の思想：API のフィールド名を ComfyUI ワークフロー JSON のノード入力に
1:1 で対応させ、バックエンドで差し込む。対応表は WORKFLOWS の ``mapping`` に持つ。

別のワークフローに差し替えたい場合は、`workflows/*.json` を置き換え、下の
``mapping`` の (node_id, input_key) を新しいノード ID に合わせて更新すればよい。
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

_WF_DIR = Path(__file__).resolve().parent / "workflows"

# ComfyUI seed は 0 .. 2^63-1 の範囲
_SEED_MAX = 2**63 - 1


@dataclass
class WorkflowSpec:
    name: str
    template_file: str
    # API フィールド名 -> (node_id, input_key)
    mapping: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # 出力(SaveImage)ノード ID。history から画像を拾う際に使う
    save_node: str = "9"

    def load_template(self) -> Dict[str, Any]:
        path = _WF_DIR / self.template_file
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


WORKFLOWS: Dict[str, WorkflowSpec] = {
    "generate": WorkflowSpec(
        name="generate",
        template_file="generate.json",
        mapping={
            "prompt": ("6", "text"),
            "negative_prompt": ("7", "text"),
            "width": ("5", "width"),
            "height": ("5", "height"),
            "steps": ("3", "steps"),
            "cfg": ("3", "cfg"),
            "seed": ("3", "seed"),
            "model": ("4", "ckpt_name"),
        },
        save_node="9",
    ),
    "edit": WorkflowSpec(
        name="edit",
        template_file="edit.json",
        mapping={
            "prompt": ("6", "text"),
            "negative_prompt": ("7", "text"),
            "strength": ("3", "denoise"),
            "steps": ("3", "steps"),
            "cfg": ("3", "cfg"),
            "seed": ("3", "seed"),
            "model": ("4", "ckpt_name"),
            # base_image は ComfyUI の /upload/image が返す name を差し込む
            "base_image": ("10", "image"),
        },
        save_node="9",
    ),
}


def resolve_seed(seed: Any) -> int:
    """None / 負値ならランダムシードを採番する。"""
    if seed is None or (isinstance(seed, int) and seed < 0):
        return random.randint(0, _SEED_MAX)
    return int(seed)


def build_workflow(kind: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """テンプレートに params を差し込んだ ComfyUI プロンプト JSON を返す。

    Returns:
        (workflow_json, resolved_params)
        resolved_params は実際に使われた値（採番後の seed 等）。ログ／再現用。

    Raises:
        KeyError: mapping が指す node_id / input_key がテンプレートに存在しない場合。
                  （ワークフロー差し替え時に mapping の更新漏れを検知するため）
    """
    if kind not in WORKFLOWS:
        raise ValueError(f"未知のワークフロー種別: {kind}")
    spec = WORKFLOWS[kind]
    wf = copy.deepcopy(spec.load_template())

    resolved = dict(params)
    resolved["seed"] = resolve_seed(params.get("seed"))

    for field_name, (node_id, input_key) in spec.mapping.items():
        if field_name not in resolved:
            continue
        value = resolved[field_name]
        # model=None などはテンプレートの既定値を尊重してスキップ
        if value is None:
            continue
        if node_id not in wf:
            raise KeyError(
                f"[{kind}] mapping のノード '{node_id}' がテンプレートに無い。"
                f" workflows/{spec.template_file} と mapping を確認"
            )
        node_inputs = wf[node_id].setdefault("inputs", {})
        if input_key not in node_inputs:
            raise KeyError(
                f"[{kind}] ノード '{node_id}' に入力 '{input_key}' が無い。"
                f" workflows/{spec.template_file} と mapping を確認"
            )
        node_inputs[input_key] = value

    return wf, resolved

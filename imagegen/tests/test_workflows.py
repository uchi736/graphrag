"""ワークフロー差し込みロジックの単体テスト（ComfyUI 不要）。"""
from __future__ import annotations

import pytest

from imagegen.backend.workflows import WORKFLOWS, build_workflow, resolve_seed


def test_generate_injection_maps_fields_to_nodes():
    params = {
        "prompt": "a red apple",
        "negative_prompt": "blurry",
        "width": 768,
        "height": 512,
        "steps": 30,
        "cfg": 6.5,
        "seed": 12345,
        "model": "my_ckpt.safetensors",
    }
    wf, resolved = build_workflow("generate", params)

    assert wf["6"]["inputs"]["text"] == "a red apple"
    assert wf["7"]["inputs"]["text"] == "blurry"
    assert wf["5"]["inputs"]["width"] == 768
    assert wf["5"]["inputs"]["height"] == 512
    assert wf["3"]["inputs"]["steps"] == 30
    assert wf["3"]["inputs"]["cfg"] == 6.5
    assert wf["3"]["inputs"]["seed"] == 12345
    assert wf["4"]["inputs"]["ckpt_name"] == "my_ckpt.safetensors"
    assert resolved["seed"] == 12345


def test_generate_model_none_keeps_template_default():
    default_ckpt = WORKFLOWS["generate"].load_template()["4"]["inputs"]["ckpt_name"]
    wf, _ = build_workflow("generate", {"prompt": "x", "model": None, "seed": 1})
    assert wf["4"]["inputs"]["ckpt_name"] == default_ckpt


def test_edit_injection_including_base_image_and_strength():
    params = {
        "prompt": "make it snowy",
        "negative_prompt": "",
        "strength": 0.4,
        "steps": 25,
        "cfg": 7.0,
        "seed": 7,
        "model": None,
        "base_image": "myupload.png",
    }
    wf, resolved = build_workflow("edit", params)

    assert wf["6"]["inputs"]["text"] == "make it snowy"
    assert wf["3"]["inputs"]["denoise"] == 0.4  # strength -> denoise
    assert wf["3"]["inputs"]["steps"] == 25
    assert wf["10"]["inputs"]["image"] == "myupload.png"


def test_resolve_seed_randomizes_on_none_or_negative():
    assert resolve_seed(42) == 42
    assert 0 <= resolve_seed(None) <= 2**63 - 1
    assert 0 <= resolve_seed(-1) <= 2**63 - 1


def test_build_workflow_does_not_mutate_template():
    t1 = WORKFLOWS["generate"].load_template()
    build_workflow("generate", {"prompt": "mutate?", "seed": 1})
    t2 = WORKFLOWS["generate"].load_template()
    assert t1["6"]["inputs"]["text"] == t2["6"]["inputs"]["text"]


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_workflow("nope", {"prompt": "x"})


def test_mapping_nodes_exist_in_templates():
    # mapping が指すノード ID は必ずテンプレートに存在すること
    for kind, spec in WORKFLOWS.items():
        template = spec.load_template()
        for field_name, (node_id, input_key) in spec.mapping.items():
            assert node_id in template, f"{kind}: node {node_id} missing"
            assert input_key in template[node_id]["inputs"], (
                f"{kind}: {node_id}.{input_key} missing"
            )
        assert spec.save_node in template

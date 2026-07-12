"""API のスモークテスト。ComfyClient を差し替えて ComfyUI 無しで検証する。"""
from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from imagegen.backend.main import create_app


class FakeComfy:
    """ComfyClient 互換の最小フェイク。実際の推論はせず即完了を返す。"""

    def __init__(self):
        self.submitted = []
        self.uploaded = []

    def ping(self):
        return True

    def list_checkpoints(self):
        return ["fake_a.safetensors", "fake_b.safetensors"]

    def upload_image(self, data, filename, overwrite=True):
        self.uploaded.append((filename, len(data)))
        return filename

    def submit(self, workflow):
        self.submitted.append(workflow)
        return "prompt-xyz"

    def wait(self, prompt_id, poll_interval, timeout, should_cancel=lambda: False, **_):
        return {"outputs": {"9": {"images": [
            {"filename": "out.png", "subfolder": "", "type": "output"}
        ]}}, "status": {"completed": True}}

    @staticmethod
    def extract_images(entry, save_node=None):
        from imagegen.backend.comfy_client import ComfyClient
        return ComfyClient.extract_images(entry, save_node)

    def download_image(self, ref):
        return b"\x89PNG\r\n\x1a\n-fake-image-bytes"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("IMAGEGEN_UPLOAD_DIR", str(tmp_path / "up"))
    # get_settings は lru_cache のためクリア
    from imagegen.backend import config
    config.get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as c:
        c.app.state.ctx.comfy = FakeComfy()  # ComfyUI を差し替え
        yield c
    config.get_settings.cache_clear()


def _wait_done(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        if job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["comfyui_reachable"] is True


def test_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()["checkpoints"] == ["fake_a.safetensors", "fake_b.safetensors"]


def test_generate_flow(client):
    r = client.post("/generate", json={"prompt": "a cat", "seed": 5})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = _wait_done(client, job_id)
    assert job["state"] == "done"
    assert job["prompt_id"] == "prompt-xyz"
    assert len(job["image_urls"]) == 1

    # 画像が配信できる
    img = client.get(job["image_urls"][0])
    assert img.status_code == 200
    assert img.content.startswith(b"\x89PNG")


def test_generate_validation_error(client):
    r = client.post("/generate", json={"prompt": ""})  # 空プロンプトは 422
    assert r.status_code == 422


def test_edit_flow(client):
    files = {"base_image": ("base.png", io.BytesIO(b"\x89PNG\r\n\x1a\nbase"), "image/png")}
    data = {"prompt": "make it night", "strength": "0.5"}
    r = client.post("/edit", files=files, data=data)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = _wait_done(client, job_id)
    assert job["state"] == "done"
    assert len(job["image_urls"]) == 1
    # アップロードされた base 画像が ComfyUI に渡っている
    assert client.app.state.ctx.comfy.uploaded


def test_edit_requires_base_image(client):
    r = client.post("/edit", data={"prompt": "x"})
    assert r.status_code == 422


def test_job_not_found(client):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_image_path_traversal_blocked(client):
    assert client.get("/images/..%2f..%2fetc%2fpasswd").status_code in (400, 404)

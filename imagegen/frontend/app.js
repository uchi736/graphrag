// 画像生成・編集 UI（素の JS）。同一オリジンのバックエンド API を叩く。
"use strict";

const POLL_MS = 1500;

// ---- タブ切替 --------------------------------------------------------------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const name = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === `panel-${name}`)
    );
  });
});

// ---- ヘルス / モデル一覧 ---------------------------------------------------
async function refreshHealth() {
  const el = document.getElementById("health");
  try {
    const r = await fetch("/health");
    const d = await r.json();
    if (d.comfyui_reachable) {
      el.textContent = "● ComfyUI 接続OK";
      el.className = "health ok";
    } else {
      el.textContent = "● ComfyUI 未接続";
      el.className = "health bad";
    }
  } catch {
    el.textContent = "● API 未接続";
    el.className = "health bad";
  }
}

async function loadModels() {
  let checkpoints = [];
  try {
    const r = await fetch("/models");
    const d = await r.json();
    checkpoints = d.checkpoints || [];
  } catch {
    /* ComfyUI 未接続時は空 */
  }
  document.querySelectorAll("[data-model-select]").forEach((sel) => {
    sel.innerHTML = "";
    const def = document.createElement("option");
    def.value = "";
    def.textContent = checkpoints.length ? "（ワークフロー既定）" : "（モデル未取得）";
    sel.appendChild(def);
    checkpoints.forEach((c) => {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      sel.appendChild(o);
    });
  });
}

// ---- ジョブのポーリング表示 ------------------------------------------------
async function pollJob(jobId, resultEl) {
  while (true) {
    let job;
    try {
      const r = await fetch(`/jobs/${jobId}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      job = await r.json();
    } catch (e) {
      renderStatus(resultEl, "failed", `状態取得に失敗: ${e.message}`);
      return;
    }
    renderJob(resultEl, job);
    if (["done", "failed", "cancelled"].includes(job.state)) return;
    await sleep(POLL_MS);
  }
}

function renderStatus(resultEl, state, text) {
  resultEl.innerHTML = `<div class="status ${state}">${text}</div>`;
}

const STATE_LABEL = {
  queued: "待機中",
  running: "生成中",
  done: "完了",
  failed: "失敗",
  cancelled: "キャンセル",
};

function renderJob(resultEl, job) {
  const label = STATE_LABEL[job.state] || job.state;
  const spinner = ["queued", "running"].includes(job.state) ? '<span class="spinner"></span> ' : "";
  let html = `<div class="status ${job.state}">${spinner}${label}（job ${job.job_id}）</div>`;

  if (job.state === "failed" && job.error) {
    html += `<div class="meta">${escapeHtml(job.error)}</div>`;
  }
  if (job.state === "done" && job.image_urls && job.image_urls.length) {
    html += job.image_urls
      .map((u) => `<a href="${u}" target="_blank" rel="noopener"><img src="${u}" alt="結果画像" /></a>`)
      .join("");
  }
  if (job.params && Object.keys(job.params).length) {
    const meta = { seed: job.params.seed, steps: job.params.steps, model: job.params.model || "(既定)" };
    html += `<div class="meta">${escapeHtml(JSON.stringify(meta))}</div>`;
  }
  resultEl.innerHTML = html;
}

// ---- 生成フォーム ----------------------------------------------------------
document.getElementById("form-generate").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const resultEl = form.parentElement.querySelector("[data-result]");
  const body = {
    prompt: form.prompt.value,
    negative_prompt: form.negative_prompt.value,
    width: Number(form.width.value),
    height: Number(form.height.value),
    steps: Number(form.steps.value),
    cfg: Number(form.cfg.value),
    seed: form.seed.value === "" ? null : Number(form.seed.value),
    model: form.model.value || null,
  };
  await submitJob(form, resultEl, () =>
    fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
});

// ---- 編集フォーム ----------------------------------------------------------
const editForm = document.getElementById("form-edit");
editForm.base_image.addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = editForm.querySelector("[data-preview]");
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  } else {
    preview.hidden = true;
  }
});

editForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const resultEl = form.parentElement.querySelector("[data-result]");
  const fd = new FormData();
  fd.append("base_image", form.base_image.files[0]);
  fd.append("prompt", form.prompt.value);
  fd.append("negative_prompt", form.negative_prompt.value);
  fd.append("strength", form.strength.value);
  fd.append("steps", form.steps.value);
  fd.append("cfg", form.cfg.value);
  if (form.seed.value !== "") fd.append("seed", form.seed.value);
  if (form.model.value) fd.append("model", form.model.value);
  await submitJob(form, resultEl, () => fetch("/edit", { method: "POST", body: fd }));
});

// ---- 共通投入処理 ----------------------------------------------------------
async function submitJob(form, resultEl, doFetch) {
  const btn = form.querySelector(".run");
  btn.disabled = true;
  renderStatus(resultEl, "queued", '<span class="spinner"></span> 投入中…');
  try {
    const r = await doFetch();
    if (!r.ok) {
      const detail = await safeDetail(r);
      renderStatus(resultEl, "failed", `投入失敗 (HTTP ${r.status}): ${detail}`);
      return;
    }
    const { job_id } = await r.json();
    await pollJob(job_id, resultEl);
  } catch (e) {
    renderStatus(resultEl, "failed", `通信エラー: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

// ---- ユーティリティ --------------------------------------------------------
async function safeDetail(r) {
  try {
    const d = await r.json();
    return typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail || d);
  } catch {
    return r.statusText;
  }
}
function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---- 初期化 ----------------------------------------------------------------
refreshHealth();
loadModels();
setInterval(refreshHealth, 15000);

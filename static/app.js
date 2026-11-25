const video = document.getElementById("webcam");
const uploadPreview = document.getElementById("upload-preview");
const fileInput = document.getElementById("file-input");
const nativeCameraInput = document.getElementById("native-camera-input");
const captureCanvas = document.getElementById("capture-canvas");
const sourceIndicator = document.getElementById("source-indicator");
const errorBanner = document.getElementById("error-banner");

const thresholdInput = document.getElementById("threshold");
const thresholdVal = document.getElementById("threshold-val");
const dilationInput = document.getElementById("dilation-ksize");
const dilationVal = document.getElementById("dilation-val");
const shiftInput = document.getElementById("shift-range");
const shiftVal = document.getElementById("shift-val");
const blockInput = document.getElementById("block-size");
const blockVal = document.getElementById("block-val");
const cInput = document.getElementById("adaptive-c");
const cVal = document.getElementById("c-val");
const minCompInput = document.getElementById("min-comp");
const minCompVal = document.getElementById("mincomp-val");
const claheInput = document.getElementById("clahe-clip");
const claheVal = document.getElementById("clahe-val");

const cropModeSelect = document.getElementById("crop-mode");
const matcherModeSelect = document.getElementById("matcher-mode");
const useBlackhatInput = document.getElementById("use-blackhat");
const useSnapshotInput = document.getElementById("use-snapshot");

const userIdInput = document.getElementById("user-id");
const displayNameInput = document.getElementById("display-name");
const sessionIdInput = document.getElementById("session-id-input");

let mediaStream = null;
let selectedFile = null;
let lastSnapshot = null; // { blob, filename, cropMode }
let activeVaultSession = null;
let vaultPollTimer = null;

// Pre-fill session_id if opened via ?session=...
const params = new URLSearchParams(window.location.search);
if (params.get("session")) {
  sessionIdInput.value = params.get("session");
}

function bindRange(input, target, suffix = "") {
  input.addEventListener("input", () => {
    target.textContent = `${input.value}${suffix}`;
  });
}

bindRange(thresholdInput, thresholdVal, "%");
bindRange(dilationInput, dilationVal);
bindRange(shiftInput, shiftVal);
bindRange(blockInput, blockVal);
bindRange(cInput, cVal);
bindRange(minCompInput, minCompVal);
bindRange(claheInput, claheVal);

function showError(msg) {
  if (!msg) {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
    return;
  }
  errorBanner.hidden = false;
  errorBanner.textContent = msg;
}

async function captureWebcamRoi() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const mode = cropModeSelect.value;

  const ctx = captureCanvas.getContext("2d");
  if (mode === "auto_palm") {
    // Send full webcam frame so Skin Segmentation & Finger Valley Detection see all fingers.
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    ctx.drawImage(video, 0, 0, vw, vh);
    const blob = await new Promise((resolve) =>
      captureCanvas.toBlob(resolve, "image/png")
    );
    return { blob, filename: "webcam_full.png", cropMode: "auto_palm", allowOverride: true };
  }

  captureCanvas.width = 256;
  captureCanvas.height = 256;
  const side = Math.floor(Math.min(vw, vh) * 0.65);
  const sx = Math.floor((vw - side) / 2);
  const sy = Math.floor((vh - side) / 2);
  ctx.drawImage(video, sx, sy, side, side, 0, 0, 256, 256);

  const blob = await new Promise((resolve) =>
    captureCanvas.toBlob(resolve, "image/png")
  );
  return { blob, filename: "webcam_roi.png", cropMode: "full", allowOverride: false };
}

document.getElementById("btn-start-cam").addEventListener("click", async () => {
  showError("");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(
      "Live webcam butuh HTTPS/localhost. Buka lewat https:// atau pakai tombol 'Kamera HP (Native)'."
    );
    return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = mediaStream;
    video.hidden = false;
    uploadPreview.hidden = true;
    selectedFile = null;
    useSnapshotInput.checked = false;
    sourceIndicator.textContent = "Sumber: Webcam aktif (crop ROI otomatis)";
    document.getElementById("btn-start-cam").disabled = true;
    document.getElementById("btn-stop-cam").disabled = false;
    document.getElementById("btn-snap").disabled = false;
  } catch (err) {
    showError(`Gagal mengakses kamera: ${err.message}`);
  }
});

document.getElementById("btn-stop-cam").addEventListener("click", () => {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  document.getElementById("btn-start-cam").disabled = false;
  document.getElementById("btn-stop-cam").disabled = true;
  document.getElementById("btn-snap").disabled = true;
  sourceIndicator.textContent = selectedFile
    ? `Sumber: File (${selectedFile.name})`
    : "Sumber: belum dipilih";
});

document.getElementById("btn-snap").addEventListener("click", async () => {
  if (!mediaStream || video.videoWidth <= 0) return;
  lastSnapshot = await captureWebcamRoi();
  useSnapshotInput.checked = true;
  uploadPreview.src = URL.createObjectURL(lastSnapshot.blob);
  uploadPreview.hidden = false;
  video.hidden = true;
  sourceIndicator.textContent = "Sumber: Snapshot dibekukan (siap uji slider)";
});

useSnapshotInput.addEventListener("change", () => {
  if (!useSnapshotInput.checked && mediaStream) {
    uploadPreview.hidden = true;
    video.hidden = false;
    sourceIndicator.textContent = "Sumber: Webcam aktif (crop ROI otomatis)";
  } else if (useSnapshotInput.checked && lastSnapshot) {
    uploadPreview.src = URL.createObjectURL(lastSnapshot.blob);
    uploadPreview.hidden = false;
    video.hidden = true;
    sourceIndicator.textContent = `Sumber: Snapshot (${lastSnapshot.filename})`;
  }
});

function handlePickedImageFile(file) {
  if (!file) return;
  selectedFile = file;
  useSnapshotInput.checked = false;
  uploadPreview.src = URL.createObjectURL(file);
  uploadPreview.hidden = false;
  video.hidden = true;
  sourceIndicator.textContent = `Sumber: Foto (${file.name || "kamera_hp.jpg"})`;
  showError("");
}

fileInput.addEventListener("change", () => handlePickedImageFile(fileInput.files[0]));
nativeCameraInput.addEventListener("change", () =>
  handlePickedImageFile(nativeCameraInput.files[0])
);

async function getInputBlobAndCropMode() {
  if (useSnapshotInput.checked && lastSnapshot) {
    return {
      ...lastSnapshot,
      cropMode: lastSnapshot.allowOverride ? cropModeSelect.value : lastSnapshot.cropMode,
    };
  }

  if (mediaStream && video.videoWidth > 0 && !video.hidden) {
    const snap = await captureWebcamRoi();
    lastSnapshot = snap;
    return snap;
  }

  if (selectedFile) {
    const snap = {
      blob: selectedFile,
      filename: selectedFile.name,
      cropMode: cropModeSelect.value,
      allowOverride: true,
    };
    lastSnapshot = snap;
    return snap;
  }

  if (lastSnapshot) {
    return {
      ...lastSnapshot,
      cropMode: lastSnapshot.allowOverride ? cropModeSelect.value : lastSnapshot.cropMode,
    };
  }

  throw new Error("Aktifkan webcam atau unggah gambar telapak tangan terlebih dahulu.");
}

function updateQualityAndStages(
  quality,
  stages,
  templateB64 = null,
  overlayB64 = null,
  orbMatchesB64 = null
) {
  const lightBadge = document.getElementById("lighting-badge");
  lightBadge.textContent = `Cahaya: ${quality.lighting_status} (${quality.mean_brightness}) | ROI: ${quality.alignment_method}`;
  lightBadge.className = "pill " + (
    quality.lighting_status === "ok" ? "pill-ok" : "pill-warn"
  );

  document.getElementById("latency-badge").textContent = `${quality.elapsed_ms} ms`;
  document.getElementById("m-pixels").textContent = quality.skeleton_pixels;

  if (stages) {
    document.getElementById("img-s0").src = stages.stage0_roi_alignment || "";
    document.getElementById("img-s1").src = stages.stage1_green_roi || "";
    document.getElementById("img-s2").src = stages.stage2_resized_256 || "";
    document.getElementById("img-s3").src = stages.stage3_clahe_median || "";
    document.getElementById("img-s4").src = stages.stage4_adaptive_binary || "";
    document.getElementById("img-s5a").src = stages.stage5a_morph_clean || "";
    document.getElementById("img-s5b").src = stages.stage5b_skeleton_1px || "";
  }
  document.getElementById("img-tmpl").src = templateB64 || "";
  document.getElementById("img-overlay").src = overlayB64 || "";
  document.getElementById("img-orb").src = orbMatchesB64 || "";
}

function setAuthBox(stateClass, title, meta) {
  const box = document.getElementById("auth-result");
  box.className = `auth-box ${stateClass}`;
  box.querySelector(".auth-status").textContent = title;
  box.querySelector(".auth-meta").textContent = meta;
}

async function callEndpoint(path, buildExtraFields) {
  showError("");
  try {
    const { blob, filename, cropMode } = await getInputBlobAndCropMode();
    const form = new FormData();
    form.append("file", blob, filename);
    form.append("crop_mode", cropMode);
    form.append("use_blackhat", String(useBlackhatInput.checked));
    form.append("clip_limit", claheInput.value);
    form.append("block_size", blockInput.value);
    form.append("c_val", cInput.value);
    form.append("min_component_px", minCompInput.value);
    buildExtraFields(form);

    const res = await fetch(path, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    return data;
  } catch (err) {
    showError(err.message);
    return null;
  }
}

document.getElementById("btn-inspect").addEventListener("click", async () => {
  const data = await callEndpoint("/api/pipeline/inspect", () => {});
  if (!data) return;
  updateQualityAndStages(data.quality, data.stages);
  setAuthBox(
    "auth-idle",
    "INSPEKSI 6 TAHAP SELESAI",
    `Density skeleton: ${(data.quality.skeleton_density * 100).toFixed(2)}% | Proses ${data.quality.elapsed_ms} ms`
  );
  document.getElementById("m-iou").textContent = "-";
  document.getElementById("m-orb").textContent = "-";
  document.getElementById("m-shift").textContent = "-";
});

const ENROLL_PROMPTS = [
  "Enroll Sampel (1/3: Posisi Normal)",
  "Enroll Sampel (2/3: Agak Miring)",
  "Enroll Sampel (3/3: Jarak Berbeda)",
];
let cachedUsers = [];

function updateEnrollWizardUI() {
  const uid = userIdInput.value.trim();
  const found = cachedUsers.find((u) => u.user_id === uid);
  const count = found ? found.sample_count : 0;
  const btn = document.getElementById("btn-enroll");
  const hint = document.getElementById("enroll-wizard-hint");
  if (count >= 3) {
    btn.textContent = "Sampel Penuh (3/3)";
    btn.disabled = true;
    hint.textContent = `User '${uid}' sudah memiliki 3/3 sampel. Klik 'Reset Enroll' jika ingin daftar ulang.`;
  } else {
    btn.disabled = false;
    btn.textContent = ENROLL_PROMPTS[count] || ENROLL_PROMPTS[0];
    hint.textContent = `Progres '${uid || "user"}': ${count}/3 sampel tersimpan.`;
  }
}

userIdInput.addEventListener("input", updateEnrollWizardUI);

document.getElementById("btn-enroll").addEventListener("click", async () => {
  const uid = userIdInput.value.trim();
  if (!uid) {
    showError("Isi User ID terlebih dahulu untuk Enroll.");
    return;
  }
  const found = cachedUsers.find((u) => u.user_id === uid);
  const mode = found && found.sample_count > 0 ? "append" : "replace";
  const data = await callEndpoint("/api/enroll", (form) => {
    form.append("user_id", uid);
    form.append("display_name", displayNameInput.value.trim() || uid);
    form.append("enroll_mode", mode);
  });
  if (!data) return;
  updateQualityAndStages(data.quality, data.stages);
  setAuthBox(
    "auth-ok",
    `TERDAFTAR (${data.user.sample_count}/3): ${data.user.display_name}`,
    `User ID '${data.user.user_id}' tersimpan (${data.user.sample_count}/3 sampel, rata-rata ${data.user.skeleton_pixels} px).`
  );
  await refreshUsers();
});

document.getElementById("btn-enroll-reset").addEventListener("click", async () => {
  const uid = userIdInput.value.trim();
  if (!uid) {
    showError("Isi User ID terlebih dahulu untuk Reset Enroll.");
    return;
  }
  const data = await callEndpoint("/api/enroll", (form) => {
    form.append("user_id", uid);
    form.append("display_name", displayNameInput.value.trim() || uid);
    form.append("enroll_mode", "replace");
  });
  if (!data) return;
  updateQualityAndStages(data.quality, data.stages);
  setAuthBox(
    "auth-ok",
    `RESET ENROLL (1/3): ${data.user.display_name}`,
    `Sampel user '${data.user.user_id}' di-reset dan disimpan ulang (1/3).`
  );
  await refreshUsers();
});

document.getElementById("btn-verify").addEventListener("click", async () => {
  const data = await callEndpoint("/api/verify", (form) => {
    const uid = userIdInput.value.trim();
    const sid = sessionIdInput.value.trim();
    if (uid) form.append("user_id", uid);
    if (sid) form.append("session_id", sid);
    form.append("matcher_mode", matcherModeSelect.value);
    form.append("threshold", String(Number(thresholdInput.value) / 100));
    form.append("dilation_ksize", dilationInput.value);
    form.append("shift_range", shiftInput.value);
    form.append("enable_alignment", String(Number(shiftInput.value) > 0));
  });
  if (!data) return;

  updateQualityAndStages(
    data.quality,
    data.stages,
    data.template_preview_b64,
    data.alignment_overlay_b64,
    data.orb_matches_b64
  );
  document.getElementById("m-iou").textContent = `${(data.match.dilated_iou * 100).toFixed(1)}%`;
  document.getElementById("m-orb").textContent = `${(data.match.orb_score * 100).toFixed(0)}% (${data.match.orb_good_matches})`;
  document.getElementById("m-shift").textContent = `(${data.match.shift_x}, ${data.match.shift_y})`;

  if (data.verified) {
    const who = data.matched_user ? `${data.matched_user.display_name} (${data.matched_user.user_id})` : "-";
    const sampleNote = `Sampel #${data.match.matched_sample_index + 1}/${data.match.samples_checked}`;
    const vaultMsg = data.vault_unlocked ? " | Vault Lintas Perangkat TERBUKA!" : "";
    setAuthBox("auth-ok", `TERVERIFIKASI (${data.match.score_percent}%)`, `User: ${who} | ${sampleNote}${vaultMsg}`);
  } else {
    setAuthBox("auth-fail", `DITOLAK (${data.match.score_percent}%)`, `Alasan: ${data.match.reason} (cek ${data.match.samples_checked} sampel)`);
  }
});

async function refreshUsers() {
  const res = await fetch("/api/users");
  if (!res.ok) return;
  const users = await res.json();
  cachedUsers = users;
  updateEnrollWizardUI();
  const ul = document.getElementById("users-list");
  ul.innerHTML = "";
  if (users.length === 0) {
    ul.innerHTML = '<li class="hint">Belum ada user terdaftar.</li>';
    return;
  }
  users.forEach((u) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span><strong>${u.user_id}</strong> (${u.display_name}) — ${u.sample_count}/3 sampel, ${u.skeleton_pixels} px</span>
      <span class="button-row" style="margin:0">
        <a href="/api/users/${encodeURIComponent(u.user_id)}/template" target="_blank" class="btn btn-small btn-secondary">Skeleton</a>
        <button data-del="${u.user_id}" class="btn btn-small btn-danger">Hapus</button>
      </span>
    `;
    li.querySelector("[data-del]").addEventListener("click", async () => {
      await fetch(`/api/users/${encodeURIComponent(u.user_id)}`, { method: "DELETE" });
      refreshUsers();
    });
    ul.appendChild(li);
  });
}

document.getElementById("btn-refresh-users").addEventListener("click", refreshUsers);

// Vault session polling
function renderVaultState(data) {
  const pill = document.getElementById("vault-pill");
  const content = document.getElementById("vault-content");
  if (data.status === "unlocked") {
    pill.textContent = "TERBUKA";
    pill.className = "pill pill-ok";
    content.className = "vault-unlocked";
    content.textContent = `🔓 Vault Terbuka oleh '${data.unlocked_by}' (Skor ${data.score_percent}%). Data rahasia dapat diakses.`;
  } else {
    pill.textContent = "TERKUNCI";
    pill.className = "pill pill-danger";
    content.className = "vault-locked";
    content.textContent = "🔒 Vault Terkunci — Menunggu verifikasi telapak tangan (polling 1 detik)...";
  }
}

async function pollVault() {
  if (!activeVaultSession) return;
  const res = await fetch(`/api/vault/status/${encodeURIComponent(activeVaultSession)}`);
  if (res.ok) {
    renderVaultState(await res.json());
  }
}

document.getElementById("btn-create-vault").addEventListener("click", async () => {
  const res = await fetch("/api/vault/session", { method: "POST" });
  if (!res.ok) return;
  const data = await res.json();
  activeVaultSession = data.session_id;
  sessionIdInput.value = data.session_id;

  let baseOrigin = window.location.origin;
  try {
    const netRes = await fetch("/api/network-info");
    if (netRes.ok) {
      const net = await netRes.json();
      if (net.mobile_url) baseOrigin = net.mobile_url.replace(/\/$/, "");
    }
  } catch {
    // Fallback to current origin if network-info check fails
  }

  const url = `${baseOrigin}/?session=${encodeURIComponent(data.session_id)}`;
  document.getElementById("vault-sid").textContent = data.session_id;
  const link = document.getElementById("vault-link");
  link.href = url;
  link.textContent = url;
  document.getElementById("vault-info").hidden = false;
  document.getElementById("btn-lock-vault").disabled = false;
  renderVaultState(data);

  if (vaultPollTimer) clearInterval(vaultPollTimer);
  vaultPollTimer = setInterval(pollVault, 1000);
});

document.getElementById("btn-lock-vault").addEventListener("click", async () => {
  if (!activeVaultSession) return;
  const res = await fetch(`/api/vault/lock/${encodeURIComponent(activeVaultSession)}`, {
    method: "POST",
  });
  if (res.ok) renderVaultState(await res.json());
});

refreshUsers();

"use strict";

const state = { user: null, csrf: null, analyses: [], currentId: null, current: null, poll: null, authMode: "login", resetToken: null };
const $ = (id) => document.getElementById(id);

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = String(content);
  return node;
}

function show(id, visible) { $(id).classList.toggle("hidden", !visible); }
function toast(message) {
  $("toast").textContent = message;
  show("toast", true);
  window.setTimeout(() => show("toast", false), 4500);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.csrf && options.method && !["GET", "HEAD"].includes(options.method.toUpperCase())) {
    headers.set("X-CSRF-Token", state.csrf);
  }
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch (_) { /* Empty or non-JSON error response. */ }
    throw new Error(message);
  }
  return response.json();
}

function requestJson(path, method, body) {
  return api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function updateAccount() {
  const signedIn = Boolean(state.user);
  show("signed-in", signedIn);
  show("signed-out", !signedIn);
  show("open-auth", !signedIn);
  show("logout", signedIn);
  show("manage-billing", signedIn && state.user.plan !== "free");
  show("email-gate", signedIn && !state.user.email_verified);
  $("upload-form").querySelector("button[type=submit]").disabled = signedIn && !state.user.email_verified;
  $("account-label").textContent = signedIn ? `${state.user.email} · ${state.user.plan}` : "";
  renderPlans();
}

function openAuth(mode) {
  state.authMode = mode;
  $("auth-title").textContent = mode === "register" ? "Create your account" : "Sign in";
  $("auth-submit").textContent = mode === "register" ? "Create account" : "Sign in";
  $("switch-auth").textContent = mode === "register" ? "Already have an account? Sign in" : "Create a free account";
  show("forgot-password", mode === "login");
  $("password").autocomplete = mode === "register" ? "new-password" : "current-password";
  $("auth-error").textContent = "";
  if (!$("auth-dialog").open) $("auth-dialog").showModal();
}

function openReset(token = null) {
  state.resetToken = token;
  const changing = Boolean(token);
  $("reset-title").textContent = changing ? "Set a new password" : "Reset password";
  $("reset-description").textContent = changing ? "Choose a new password with at least 12 characters." :
    "Enter your account email. We will send a one-time recovery link.";
  show("reset-email-label", !changing);
  show("reset-password-label", changing);
  $("reset-email").required = !changing;
  $("new-password").required = changing;
  $("reset-submit").textContent = changing ? "Set new password" : "Send recovery link";
  $("reset-error").textContent = "";
  if (!$("reset-dialog").open) $("reset-dialog").showModal();
}

async function refreshSession() {
  try {
    state.user = await api("/v1/auth/me");
    state.csrf = state.user.csrf_token || null;
    updateAccount();
    await Promise.all([loadAnalyses(), loadUsage()]);
  } catch (_) {
    state.user = null;
    state.csrf = null;
    updateAccount();
  }
}

async function loadUsage() {
  if (!state.user) return;
  try {
    const usage = await api("/v1/account/usage");
    $("usage-label").textContent = `${usage.used_minutes + usage.reserved_minutes} of ${usage.monthly_limit} minutes this month`;
  } catch (error) { $("usage-label").textContent = error.message; }
}

function formatDate(value) {
  if (!value) return "";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function renderAnalyses() {
  const list = $("analysis-list");
  list.replaceChildren();
  if (!state.analyses.length) {
    list.append(element("p", "empty-state", "No analyses yet. Upload a video to get started."));
    return;
  }
  for (const job of state.analyses) {
    const button = element("button", "analysis-item" + (job.id === state.currentId ? " active" : ""));
    button.type = "button";
    const left = element("span");
    left.append(element("strong", "", job.recipe.replaceAll("_", " ")),
                element("small", "", formatDate(job.created_at)));
    button.append(left, element("span", "state", job.status));
    button.addEventListener("click", () => selectAnalysis(job.id));
    list.append(button);
  }
}

async function loadAnalyses() {
  if (!state.user) return;
  try {
    state.analyses = (await api("/v1/analyses")).analyses || [];
    renderAnalyses();
  } catch (error) { toast(error.message); }
}

function stopPolling() {
  if (state.poll) window.clearInterval(state.poll);
  state.poll = null;
}

async function selectAnalysis(id) {
  stopPolling();
  state.currentId = id;
  show("result", true);
  renderAnalyses();
  await refreshCurrent();
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function refreshCurrent() {
  if (!state.currentId) return;
  try {
    const job = await api(`/v1/analyses/${state.currentId}`);
    state.current = job;
    $("result-title").textContent = job.recipe.replaceAll("_", " ");
    $("result-status").textContent = `${job.status} · ${formatDate(job.created_at)}`;
    const working = ["uploading", "queued", "processing"].includes(job.status);
    show("progress-area", working);
    show("result-content", job.status === "completed");
    $("open-report").disabled = job.status !== "completed";
    show("annotated-video", job.status === "completed" && job.recipe === "event_pulse");
    $("annotated-video").href = `/v1/analyses/${encodeURIComponent(job.id)}/artifacts/output.mp4`;
    $("progress-fill").style.width = `${job.progress || 0}%`;
    $("progress-label").textContent = `${job.stage || job.status} · ${job.progress || 0}%`;
    show("cancel-analysis", working);
    if (job.status === "completed") {
      stopPolling();
      await loadResult(job.id);
    } else if (working && !state.poll) {
      state.poll = window.setInterval(() => refreshCurrent(), 2500);
    } else if (!working) {
      stopPolling();
      if (job.error_message) toast(job.error_message);
    }
    await loadAnalyses();
    await loadUsage();
  } catch (error) { stopPolling(); toast(error.message); }
}

function metric(label, value) {
  const card = element("div", "metric-card");
  card.append(element("small", "", label), element("strong", "", value));
  return card;
}

async function loadResult(id) {
  const [overview, momentData] = await Promise.all([
    api(`/v1/analyses/${id}/overview`), api(`/v1/analyses/${id}/moments`),
  ]);
  if (id !== state.currentId) return;
  const counts = overview.stats?.people_count || {};
  $("metric-cards").replaceChildren(
    metric("Unique tracks", overview.tracks_summary?.unique_people ?? "—"),
    metric("Peak visible people", counts.max ?? "—"),
    metric("Key moments", (momentData.moments || []).length),
  );
  const list = $("moment-list");
  list.replaceChildren();
  const moments = momentData.moments || [];
  if (!moments.length) list.append(element("p", "empty-state", "No notable moments were detected. Review the overview and source video."));
  for (const moment of moments) {
    const button = element("button", "moment");
    button.type = "button";
    const thumb = element("span", "moment-placeholder", "Video");
    const content = element("span");
    content.append(element("strong", "", moment.title),
                   element("small", "", `${formatTime(moment.start_sec)}–${formatTime(moment.end_sec)}`));
    button.append(thumb, content);
    button.addEventListener("click", () => showMoment(id, moment));
    list.append(button);
    if (moment.evidence_ids?.length) {
      const evidenceId = moment.evidence_ids[0];
      const image = element("img");
      image.alt = `Keyframe at ${formatTime(moment.start_sec)}`;
      image.loading = "lazy";
      image.src = `/v1/analyses/${encodeURIComponent(id)}/evidence/${encodeURIComponent(evidenceId)}/image`;
      image.addEventListener("load", () => thumb.replaceWith(image));
    }
  }
  $("source-video").src = `/v1/analyses/${encodeURIComponent(id)}/video`;
}

function showMoment(id, moment) {
  if (id !== state.currentId) return;
  show("video-wrap", true);
  $("video-caption").textContent = `${moment.title} · ${formatTime(moment.start_sec)}–${formatTime(moment.end_sec)}. Tracking quality is a heuristic, not a probability.`;
  const end = Math.min(moment.end_sec, moment.start_sec + 30);
  $("clip-link").href = `/v1/analyses/${encodeURIComponent(id)}/clip?start_sec=${moment.start_sec}&end_sec=${end}`;
  show("clip-link", end > moment.start_sec);
  const video = $("source-video");
  const seek = () => { video.currentTime = Math.max(0, moment.start_sec); };
  if (video.readyState >= 1) seek(); else video.addEventListener("loadedmetadata", seek, { once: true });
  $("video-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderAnswer(result) {
  const box = $("answer");
  box.replaceChildren(element("p", "", result.answer));
  for (const item of result.evidence || []) {
    const source = element("div", "answer-source");
    const at = item.time_sec == null ? "Aggregate data" : formatTime(item.time_sec);
    source.textContent = `${at} · ${(item.sources || []).join(", ")}`;
    box.append(source);
  }
  if (result.evidence_quality === "none") box.append(element("div", "answer-source", "No verified supporting data in this analysis."));
}

async function renderPlans() {
  const target = $("plan-cards");
  if (!target) return;
  try {
    const response = await api("/v1/plans");
    target.replaceChildren();
    const footer = $("footer-links");
    footer.replaceChildren();
    const docs = element("a", "", "API documentation"); docs.href = "/docs"; footer.append(docs);
    for (const [label, url] of [["Terms", response.legal?.terms_url], ["Privacy", response.legal?.privacy_url]]) {
      if (url?.startsWith("https://")) {
        const link = element("a", "", label); link.href = url; link.rel = "noopener noreferrer";
        footer.append(link);
      }
    }
    const descriptions = {
      free: "Explore the workflow on shorter videos.",
      developer: "More room for regular analysis and API access.",
      team: "Higher monthly capacity for one account.",
    };
    for (const plan of response.plans) {
      const card = element("article", "plan-card" + (plan.id === "developer" ? " featured" : ""));
      card.append(element("h3", "", plan.id[0].toUpperCase() + plan.id.slice(1)),
                  element("p", "", descriptions[plan.id] || ""),
                  element("strong", "", plan.display_price || "Price pending"),
                  element("small", "", `${plan.monthly_minutes} analysis minutes per month`));
      const button = element("button", "button " + (plan.id === "developer" ? "button-primary" : "button-light"));
      button.type = "button";
      button.textContent = plan.id === "free" ? "Get started" : plan.checkout_available ? "Choose plan" : "Coming soon";
      button.disabled = plan.id !== "free" && !plan.checkout_available;
      button.addEventListener("click", async () => {
        if (!state.user) { openAuth("register"); return; }
        if (plan.id === "free") { $("workspace").scrollIntoView({ behavior: "smooth" }); return; }
        try {
          const result = await requestJson("/v1/billing/checkout", "POST", { plan: plan.id });
          window.location.assign(result.checkout_url);
        } catch (error) { toast(error.message); }
      });
      card.append(button);
      const details = element("ul");
      for (const detail of [`Up to ${plan.max_video_minutes} minutes per video`, `${plan.max_upload_mb} MB per upload`, `${plan.storage_gb} GB source-video storage`, `${plan.retention_days}-day retention`, plan.id === "free" ? "Web access" : "Web and API access"]) details.append(element("li", "", detail));
      card.append(details);
      target.append(card);
    }
  } catch (error) { target.replaceChildren(element("p", "empty-state", error.message)); }
}

$("open-auth").addEventListener("click", () => openAuth("login"));
$("gate-sign-in").addEventListener("click", () => openAuth("register"));
$("close-auth").addEventListener("click", () => $("auth-dialog").close());
$("switch-auth").addEventListener("click", () => openAuth(state.authMode === "login" ? "register" : "login"));
$("forgot-password").addEventListener("click", () => { $("auth-dialog").close(); openReset(); });
$("close-reset").addEventListener("click", () => $("reset-dialog").close());
$("reset-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("reset-error").textContent = "";
  try {
    if (state.resetToken) {
      await requestJson("/v1/auth/reset-password", "POST", {
        token: state.resetToken, password: $("new-password").value,
      });
      state.resetToken = null;
      window.history.replaceState({}, "", "/");
      $("reset-dialog").close();
      toast("Password updated. Sign in with your new password.");
      openAuth("login");
    } else {
      const result = await requestJson("/v1/auth/request-password-reset", "POST", {
        email: $("reset-email").value,
      });
      $("reset-dialog").close(); toast(result.message);
    }
  } catch (error) { $("reset-error").textContent = error.message; }
});
$("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("auth-error").textContent = "";
  try {
    const result = await requestJson(`/v1/auth/${state.authMode}`, "POST", {
      email: $("email").value, password: $("password").value,
    });
    if (result.message) {
      $("auth-dialog").close();
      $("password").value = "";
      toast(result.message);
      return;
    }
    state.user = result;
    state.csrf = result.csrf_token;
    $("auth-dialog").close();
    $("password").value = "";
    updateAccount();
    await Promise.all([loadAnalyses(), loadUsage()]);
    $("workspace").scrollIntoView({ behavior: "smooth" });
  } catch (error) { $("auth-error").textContent = error.message; }
});
$("logout").addEventListener("click", async () => {
  try { await api("/v1/auth/logout", { method: "POST" }); } catch (error) { toast(error.message); return; }
  stopPolling(); state.user = null; state.csrf = null; state.currentId = null; state.analyses = [];
  show("result", false); show("video-wrap", false); $("source-video").removeAttribute("src");
  updateAccount();
});
$("manage-billing").addEventListener("click", async () => {
  try {
    const result = await api("/v1/billing/portal", { method: "POST" });
    window.location.assign(result.portal_url);
  } catch (error) { toast(error.message); }
});
$("resend-email").addEventListener("click", async () => {
  try { await api("/v1/auth/resend-verification", { method: "POST" }); toast("Verification link sent."); }
  catch (error) { toast(error.message); }
});
const regionState = { zones: [], lines: [], draft: [], mode: "zone", previewUrl: null, ready: false };

function regionMessage(message) { $("region-status").textContent = message; }

function paintRegions() {
  const canvas = $("region-canvas");
  if (!regionState.ready) return;
  const context = canvas.getContext("2d");
  context.drawImage($("region-video"), 0, 0, canvas.width, canvas.height);
  const drawPath = (points, color, closed) => {
    if (!points.length) return;
    context.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) context.moveTo(x * canvas.width, y * canvas.height);
      else context.lineTo(x * canvas.width, y * canvas.height);
    });
    if (closed) context.closePath();
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.stroke();
    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x * canvas.width, y * canvas.height, 4, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }
  };
  regionState.zones.forEach((zone) => drawPath(zone.points, "#d5ed8c", true));
  regionState.lines.forEach((line) => drawPath([line.start, line.end], "#7ec8ff", false));
  drawPath(regionState.draft, regionState.mode === "line" ? "#7ec8ff" : "#d5ed8c", false);
}

function resetRegionPreview(file) {
  if (regionState.previewUrl) URL.revokeObjectURL(regionState.previewUrl);
  regionState.zones = [];
  regionState.lines = [];
  regionState.draft = [];
  regionState.mode = "zone";
  regionState.ready = false;
  $("region-canvas").classList.add("hidden");
  $("file-name").textContent = file?.name || "or drag it here";
  const video = $("region-video");
  video.removeAttribute("src");
  video.load();
  regionState.previewUrl = file ? URL.createObjectURL(file) : null;
  if (regionState.previewUrl) video.src = regionState.previewUrl;
  regionMessage(file ? "Preparing a frame preview…" : "Choose a video to mark a zone or line.");
}

$("region-video").addEventListener("loadeddata", () => {
  const video = $("region-video");
  const canvas = $("region-canvas");
  const scale = 640 / Math.max(video.videoWidth, video.videoHeight);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  regionState.ready = true;
  canvas.classList.remove("hidden");
  paintRegions();
  regionMessage("Click at least three points, then save the zone. Or choose Draw line and click two endpoints.");
});
$("region-video").addEventListener("error", () => regionMessage("This browser could not preview the video. You can still upload the MP4."));
$("video-file").addEventListener("change", () => resetRegionPreview($("video-file").files[0]));
$("draw-zone").addEventListener("click", () => {
  regionState.mode = "zone";
  regionState.draft = [];
  paintRegions();
  regionMessage("Click three or more points around an area, then Save zone.");
});
$("finish-zone").addEventListener("click", () => {
  if (regionState.draft.length < 3 || regionState.mode !== "zone") {
    regionMessage("A zone needs at least three points. Choose Draw zone and try again.");
    return;
  }
  if (regionState.zones.length >= 5) { regionMessage("You can draw up to five zones here."); return; }
  regionState.zones.push({ id: `zone${regionState.zones.length + 1}`, points: regionState.draft });
  regionState.draft = [];
  paintRegions();
  regionMessage(`${regionState.zones.length} zone(s), ${regionState.lines.length} line(s) saved for this upload.`);
});
$("draw-line").addEventListener("click", () => {
  regionState.mode = "line";
  regionState.draft = [];
  paintRegions();
  regionMessage("Click two points to draw a crossing line. Its direction follows the order of your clicks.");
});
$("clear-regions").addEventListener("click", () => {
  regionState.zones = [];
  regionState.lines = [];
  regionState.draft = [];
  paintRegions();
  regionMessage("Zones and lines cleared.");
});
$("region-canvas").addEventListener("pointerdown", (event) => {
  if (!regionState.ready) return;
  const bounds = event.currentTarget.getBoundingClientRect();
  const point = [
    Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
    Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
  ].map((value) => Number(value.toFixed(4)));
  if (regionState.mode === "zone" && regionState.draft.length >= 32) {
    regionMessage("A zone can have at most 32 points. Save it or clear the drawing.");
    return;
  }
  regionState.draft.push(point);
  if (regionState.mode === "line" && regionState.draft.length === 2) {
    if (regionState.lines.length >= 5) { regionState.draft = []; regionMessage("You can draw up to five lines here."); return; }
    const [start, end] = regionState.draft;
    if (start[0] !== end[0] || start[1] !== end[1]) {
      regionState.lines.push({ id: `line${regionState.lines.length + 1}`, start, end });
      regionMessage(`${regionState.zones.length} zone(s), ${regionState.lines.length} line(s) saved for this upload.`);
    }
    regionState.draft = [];
  }
  paintRegions();
});
const dropzone = document.querySelector(".dropzone");
dropzone.addEventListener("dragover", (event) => event.preventDefault());
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer.files.length) {
    $("video-file").files = event.dataTransfer.files;
    resetRegionPreview(event.dataTransfer.files[0]);
  }
});
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("video-file").files[0];
  if (!file) return;
  const button = $("upload-form").querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const params = new URLSearchParams({ recipe: $("recipe").value });
    if (regionState.zones.length) params.set("zones", JSON.stringify(regionState.zones));
    if (regionState.lines.length) params.set("lines", JSON.stringify(regionState.lines));
    const result = await api(`/v1/analyses?${params}`, {
      method: "POST", headers: { "Content-Type": "video/mp4" }, body: file,
    });
    await loadAnalyses(); await loadUsage(); await selectAnalysis(result.analysis_id);
    $("upload-form").reset(); resetRegionPreview(null);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
});
$("refresh-list").addEventListener("click", () => loadAnalyses());
$("cancel-analysis").addEventListener("click", async () => {
  try { await api(`/v1/analyses/${state.currentId}/cancel`, { method: "POST" }); await refreshCurrent(); }
  catch (error) { toast(error.message); }
});
$("delete-analysis").addEventListener("click", async () => {
  if (!state.currentId || !window.confirm("Delete this analysis and its stored video? This cannot be undone.")) return;
  try {
    await api(`/v1/analyses/${state.currentId}`, { method: "DELETE" });
    stopPolling(); state.currentId = null; state.current = null;
    show("result", false); show("video-wrap", false); $("source-video").removeAttribute("src");
    await Promise.all([loadAnalyses(), loadUsage()]); toast("Analysis deleted.");
  } catch (error) { toast(error.message); }
});
$("open-report").addEventListener("click", () => {
  if (state.currentId) window.location.assign(`/v1/analyses/${state.currentId}/report`);
});
$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.currentId) return;
  try {
    const answer = await requestJson(`/v1/analyses/${state.currentId}/query`, "POST", { question: $("question").value });
    renderAnswer(answer);
  } catch (error) { toast(error.message); }
});

async function start() {
  const token = new URLSearchParams(window.location.search).get("token");
  if (window.location.pathname === "/reset" && token) openReset(token);
  if (window.location.pathname === "/verify" && token) {
    try {
      await requestJson("/v1/auth/verify-email", "POST", { token });
      toast("Email verified. You can now use your workspace.");
    } catch (error) { toast(error.message); }
    window.history.replaceState({}, "", "/#workspace");
  }
  await refreshSession();
}
start();

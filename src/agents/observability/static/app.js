"use strict";

const state = {
  runs: [],
  selected: null,    // run_id
  trace: null,
  es: null,          // EventSource for live streaming
  llmIndex: {},      // key -> full llm call object (for the drawer)
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function fmtTime(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

// ---------- API ----------
async function fetchRuns() {
  const r = await fetch("/api/runs");
  const data = await r.json();
  state.runs = data.runs || [];
  renderRuns();
}

async function fetchTrace(runId) {
  const r = await fetch(`/api/runs/${runId}`);
  if (!r.ok) return null;
  return await r.json();
}

async function launchRun() {
  const q = $("#query").value.trim();
  if (!q) return;
  $("#launch-btn").disabled = true;
  try {
    const r = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    const data = await r.json();
    if (data.run_id) {
      await fetchRuns();
      selectRun(data.run_id, true);
    }
  } catch (e) {
    alert("Failed to launch run: " + e);
  } finally {
    $("#launch-btn").disabled = false;
  }
}

// ---------- Sidebar ----------
function renderRuns() {
  const box = $("#runs");
  box.innerHTML = "";
  if (!state.runs.length) {
    box.appendChild(el("div", "empty", "No runs yet."));
    return;
  }
  for (const run of state.runs) {
    const item = el("div", "run-item" + (run.run_id === state.selected ? " active" : ""));
    const c = run.counts || {};
    item.innerHTML = `
      <div class="q">${esc(run.query || "(no query)")}</div>
      <div class="meta">
        <span class="pill ${run.status}">${run.status}</span>
        <span>${c.iterations || 0} iter</span>
        <span>${c.actions || 0} act</span>
        ${run.has_llm_io ? '<span title="full LLM I/O captured">◆ I/O</span>' : ""}
      </div>`;
    item.onclick = () => selectRun(run.run_id, run.status === "running");
    box.appendChild(item);
  }
}

// ---------- Selection + live streaming ----------
function closeStream() {
  if (state.es) { state.es.close(); state.es = null; }
}

async function selectRun(runId, live) {
  closeStream();
  state.selected = runId;
  renderRuns();
  const t = await fetchTrace(runId);
  if (t) { state.trace = t; renderTrace(t); }
  if (live || (t && t.status === "running")) startStream(runId);
}

function startStream(runId) {
  closeStream();
  const es = new EventSource(`/api/runs/${runId}/stream`);
  state.es = es;
  es.onmessage = (ev) => {
    try {
      const t = JSON.parse(ev.data);
      if (state.selected === runId) { state.trace = t; renderTrace(t); }
    } catch {}
  };
  es.addEventListener("done", () => { closeStream(); fetchRuns(); });
  es.onerror = () => { /* keep retrying via browser default */ };
}

// ---------- Trace rendering ----------
function eventsByName(events, name) {
  return events.filter((e) => e.event === name);
}
function lastEvent(events, name) {
  const m = eventsByName(events, name);
  return m.length ? m[m.length - 1] : null;
}

function renderTrace(t) {
  state.llmIndex = {};
  const main = $("#main");
  main.innerHTML = "";

  // Header
  const head = el("div", "run-head");
  const liveDot = t.status === "running" ? '<span class="live-dot"></span>' : "";
  const c = t.counts || {};
  head.innerHTML = `
    <h2>${liveDot}${esc(t.query || "(no query)")}</h2>
    <div class="stats">
      <span class="pill ${t.status}">${t.status}</span>
      <span><b>${t.iterations.length}</b> iterations</span>
      <span><b>${c.actions || 0}</b> actions</span>
      <span><b>${c.answers || 0}</b> answers</span>
      <span><b>${(t.llm_calls || []).length}</b> LLM calls</span>
      <span class="tag mono">${esc(t.run_id)}</span>
      <span class="tag">${fmtTime(t.start_ts)}</span>
    </div>`;
  if (!t.has_llm_io) {
    head.appendChild(el("div", "tag", "⚠ Full LLM I/O was not captured for this run (older run). Flow is shown from the event trace."));
  }
  main.appendChild(head);

  // Model failover flow (run-level)
  if (t.model_flow && (t.model_flow.attempts || []).length) {
    main.appendChild(renderModelFlowPanel(t.model_flow));
  }

  // Setup phase (memory created etc.)
  if (t.setup_llm_calls && t.setup_llm_calls.length) {
    const s = el("div", "iter");
    s.appendChild(el("div", "iter-head", "<span>Setup</span>"));
    const body = el("div", "iter-body");
    body.appendChild(renderLlmChips(t.setup_llm_calls));
    s.appendChild(body);
    main.appendChild(s);
  }

  // Iterations
  for (const it of t.iterations) main.appendChild(renderIteration(it));
}

function renderIteration(it) {
  const wrap = el("div", "iter");
  const llmCount = (it.llm_calls || []).length;
  const head = el("div", "iter-head");
  head.innerHTML = `<span>Iteration ${it.iteration}</span>
    <span class="badge-grp">${llmCount ? `<span class="tag">${llmCount} LLM call${llmCount > 1 ? "s" : ""}</span>` : ""}</span>`;
  wrap.appendChild(head);

  const body = el("div", "iter-body");
  const ev = it.events;

  // PERCEPTION
  const perc = lastEvent(ev, "perception_complete");
  const decomp = lastEvent(ev, "goals_decomposed");
  const selected = lastEvent(ev, "goal_selected");
  if (perc || decomp) {
    const goals = (perc && perc.goals) || (decomp && decomp.goals) || [];
    const selId = selected ? selected.goal_id : null;
    const block = el("div", "phase perception");
    block.appendChild(el("div", "phase-label", "Perception · goals"));
    const ul = el("ul", "goal-list");
    for (const g of goals) {
      const li = el("li", g.id === selId ? "selected" : "");
      const mark = g.done ? '<span class="check">✓</span>' : '<span class="open">○</span>';
      li.innerHTML = `${mark}${esc(g.text)} ${g.id === selId ? '<span class="tag">← selected</span>' : ""}`;
      ul.appendChild(li);
    }
    block.appendChild(ul);
    body.appendChild(block);
  }

  // DECISION
  const dec = lastEvent(ev, "decision_complete");
  if (dec) {
    const block = el("div", "phase decision");
    block.appendChild(el("div", "phase-label", "Decision"));
    const what = dec.is_answer ? "Produce answer" : `Call tool <b>${esc(dec.tool_name)}</b>`;
    block.appendChild(el("div", "kv", what));
    body.appendChild(block);
  }

  // ACTION
  const aStart = lastEvent(ev, "action_start");
  const aDone = lastEvent(ev, "action_complete");
  if (aStart || aDone) {
    const block = el("div", "phase action");
    block.appendChild(el("div", "phase-label", "Action"));
    if (aStart) {
      block.appendChild(el("div", "kv", `Tool: <b>${esc(aStart.tool)}</b>`));
      block.appendChild(el("div", "preview", esc(JSON.stringify(aStart.arguments, null, 2))));
    }
    if (aDone) {
      if (aDone.artifact_id) block.appendChild(el("div", "kv", `Artifact: <b>${esc(aDone.artifact_id)}</b>`));
      block.appendChild(el("div", "preview", esc(aDone.result_preview || "")));
    }
    body.appendChild(block);
  }

  // ANSWER
  const ans = lastEvent(ev, "answer_produced");
  if (ans) {
    const block = el("div", "phase answer");
    block.appendChild(el("div", "phase-label", "Answer"));
    block.appendChild(el("div", "preview", esc(ans.answer_preview || "")));
    body.appendChild(block);
  }

  // LLM chips
  if (llmCount) body.appendChild(renderLlmChips(it.llm_calls));

  // Raw events (collapsible)
  const raw = el("details", "param");
  raw.appendChild(el("summary", null, `raw events (${ev.length})`));
  raw.appendChild(el("div", "block mono", esc(JSON.stringify(ev, null, 2))));
  body.appendChild(raw);

  wrap.appendChild(body);
  return wrap;
}

function renderLlmChips(calls) {
  const wrap = el("div");
  wrap.appendChild(el("div", "phase-label", "LLM calls"));
  for (const call of calls) {
    const key = `${state.selected}:${call.iteration}:${call.seq}`;
    state.llmIndex[key] = call;
    const resp = call.response || {};
    const tok = resp.input_tokens != null ? `${resp.input_tokens}→${resp.output_tokens ?? "?"}` : "";
    const fo = call.failover || {};
    const nAtt = (fo.worker_attempts || []).length + (fo.router_attempts || []).length;
    const nErr = [...(fo.worker_attempts || []), ...(fo.router_attempts || [])]
      .filter((a) => a.outcome === "error" || a.status === "error").length;
    const nSkip = [...(fo.worker_attempts || []), ...(fo.router_attempts || [])]
      .filter((a) => a.outcome === "skipped" || a.status === "skipped").length;
    // Only interesting when the pick wasn't a clean first-try (i.e. retries/skips).
    const showFlow = nAtt > 1 || nErr > 0 || nSkip > 0;
    const chip = el("div", "llm-chip");
    chip.innerHTML = `<span class="label">${esc(call.call_label)}</span>
      <span class="mdl">${esc(resp.model || call.request?.model || "?")}</span>
      ${tok ? `<span class="tok">${tok} tok</span>` : ""}
      ${showFlow ? `<span class="flow-badge" title="${nAtt} model attempts (${nErr} err, ${nSkip} skipped)">⚡${nAtt}${nErr ? ` <span class="fb-err">${nErr}✗</span>` : ""}</span>` : ""}
      ${call.error ? '<span class="pill error">err</span>' : ""}`;
    chip.onclick = () => openDrawer(key);
    wrap.appendChild(chip);
  }
  return wrap;
}

// ---------- Drawer: full LLM I/O ----------
function blockOf(title, content, cls, copyable) {
  const sec = el("div", "io-section");
  const h = el("div", "h");
  h.innerHTML = `<span>${esc(title)}</span>`;
  if (copyable) {
    const btn = el("button", "ghost copy", "copy");
    btn.style.padding = "1px 8px"; btn.style.fontSize = "11px";
    btn.onclick = () => navigator.clipboard.writeText(content);
    h.appendChild(btn);
  }
  sec.appendChild(h);
  sec.appendChild(el("div", "block " + (cls || ""), esc(content)));
  return sec;
}

function renderInput(req) {
  // Prefer messages array; fall back to prompt string.
  if (Array.isArray(req.messages)) {
    return req.messages.map((m) => `[${m.role}]\n${typeof m.content === "string" ? m.content : JSON.stringify(m.content, null, 2)}`).join("\n\n");
  }
  return req.prompt != null ? String(req.prompt) : "(no prompt / messages)";
}

function asText(v) {
  if (v == null) return "";
  return typeof v === "string" ? v : JSON.stringify(v, null, 2);
}

function openDrawer(key) {
  const call = state.llmIndex[key];
  if (!call) return;
  const req = call.request || {};
  const resp = call.response || {};
  $("#drawer-title").textContent = call.call_label + (call.iteration ? `  ·  iter ${call.iteration}` : "");
  const body = $("#drawer-body");
  body.innerHTML = "";

  // Meta
  const rd = resp.router_decision || {};
  const meta = el("div", "io-meta");
  meta.innerHTML = `
    <span>provider <b>${esc(resp.provider || req.provider || "?")}</b></span>
    <span>model <b>${esc(resp.model || req.model || "?")}</b></span>
    <span>tokens <b>${resp.input_tokens ?? "?"} → ${resp.output_tokens ?? "?"}</b></span>
    <span>latency <b>${call.latency_ms ?? "?"} ms</b></span>
    ${rd.tier ? `<span>tier <b>${esc(rd.tier)}</b></span>` : ""}
    ${req.auto_route ? `<span>auto_route <b>${esc(req.auto_route)}</b></span>` : ""}`;
  body.appendChild(meta);

  if (call.error) body.appendChild(blockOf("Error", call.error, "err"));

  // INPUT
  if (req.system != null) body.appendChild(blockOf("System prompt", asText(req.system), "sys", true));
  body.appendChild(blockOf("Input (messages / prompt)", renderInput(req), "usr", true));

  // OUTPUT
  body.appendChild(blockOf("Output · raw text", asText(resp.text), "out", true));
  if (resp.parsed != null) body.appendChild(blockOf("Output · parsed JSON", asText(resp.parsed), "out", true));
  if (resp.tool_calls != null && resp.tool_calls.length > 0)
    body.appendChild(blockOf("Output · tool calls", asText(resp.tool_calls), "out", true));

  // Params + extras
  const params = {
    temperature: req.temperature, max_tokens: req.max_tokens, reasoning: req.reasoning,
    cache_system: req.cache_system, tool_choice: req.tool_choice,
    provider_requested: req.provider, model_requested: req.model, auto_route: req.auto_route,
  };
  // Model failover flow for THIS call (router classifier + worker pick/retries)
  const fo = call.failover || {};
  const routerAtts = fo.router_attempts || [];
  const workerAtts = fo.worker_attempts || resp.attempted || [];
  if (routerAtts.length || workerAtts.length) {
    const sec = el("div", "io-section");
    sec.appendChild(el("div", "h", "<span>Model failover flow</span>"));
    if (routerAtts.length) {
      sec.appendChild(el("div", "flow-subhead", "Router (tier classifier)"));
      routerAtts.forEach((a) => sec.appendChild(renderAttemptRow(a, "router")));
    }
    if (workerAtts.length) {
      sec.appendChild(el("div", "flow-subhead", "Worker (model ring)"));
      workerAtts.forEach((a) => sec.appendChild(renderAttemptRow(a, "worker")));
    }
    body.appendChild(sec);
  }

  body.appendChild(detailsOf("Parameters", JSON.stringify(params, null, 2)));
  if (req.response_format != null) body.appendChild(detailsOf("response_format (schema)", asText(req.response_format)));
  if (req.tools != null) body.appendChild(detailsOf("tools", asText(req.tools)));
  if (rd && Object.keys(rd).length) body.appendChild(detailsOf("router_decision (raw)", asText(rd)));

  showDrawer(true);
}

function detailsOf(title, content) {
  const d = el("details", "param");
  d.appendChild(el("summary", null, esc(title)));
  d.appendChild(el("div", "block mono", esc(content)));
  return d;
}

function showDrawer(open) {
  $("#drawer").classList.toggle("open", open);
  $("#overlay").classList.toggle("open", open);
}

// ---------- Model failover flow rendering ----------
const OUTCOME_CLASS = {
  success: "oc-success", error: "oc-error", skipped: "oc-skipped",
  selected: "oc-selected", unparseable: "oc-warn",
};

function getOutcome(a) {
  if (a.outcome) return a.outcome;
  return ({ ok: "success", error: "error", skipped: "skipped", unparseable: "unparseable" })[a.status] || a.status || "unknown";
}

function fmtRate(r) {
  if (!r) return "";
  const p = [];
  if (r.rpm_limit != null) p.push(`rpm ${r.rpm_used ?? "?"}/${r.rpm_limit}`);
  if (r.rpd_limit != null) p.push(`rpd ${r.rpd_used ?? "?"}/${r.rpd_limit}`);
  if (r.cooldown_remaining > 0) p.push(`cd ${Number(r.cooldown_remaining).toFixed(1)}s`);
  if (r.backoff_remaining > 0) p.push(`backoff ${Number(r.backoff_remaining).toFixed(0)}s`);
  return p.join(" · ");
}

function renderAttemptRow(a) {
  const oc = getOutcome(a);
  const row = el("div", "flow-row " + (OUTCOME_CLASS[oc] || ""));
  const strat = a.strategy && a.strategy !== "router" ? a.strategy : "";
  const slot = a.slot_index != null ? `#${a.slot_index}` : "";
  const badge = strat ? `<span class="flow-strat">${esc(strat)}${slot}</span>` : "";
  const lat = a.latency_ms != null ? `<span class="flow-lat">${a.latency_ms}ms</span>` : "";
  const rate = a.rate ? `<span class="flow-rate">${esc(fmtRate(a.rate))}</span>` : "";
  const tier = a.tier ? ` → ${esc(a.tier)}` : "";
  row.innerHTML = `
    <span class="flow-dot"></span>
    <span class="flow-oc">${esc(oc)}</span>
    ${badge}
    <span class="flow-model"><b>${esc(a.provider || "?")}</b> ${esc(a.model || "")}</span>
    <span class="flow-reason">${esc(a.reason || a.error || "")}${tier}</span>
    ${lat}${rate}`;
  return row;
}

function renderModelFlowPanel(mf) {
  const c = mf.counts || {};
  const wrap = el("div", "iter mflow");
  const head = el("div", "iter-head");
  head.innerHTML = `<span>Model Flow</span>
    <span class="badge-grp">
      <span class="tag">${c.total_attempts || 0} attempts</span>
      <span class="tag">${c.worker || 0} worker · ${c.router || 0} router</span>
      <span class="tag ok">${c.success || 0}✓</span>
      <span class="tag err">${c.error || 0}✗</span>
      <span class="tag skip">${c.skipped || 0}∅</span>
    </span>`;
  wrap.appendChild(head);
  const bodyEl = el("div", "iter-body");

  // Cooldown violations
  if ((mf.violations || []).length) {
    const v = el("div", "mflow-violations");
    v.appendChild(el("div", "phase-label", `⚠ Cooldown violations (${mf.violations.length})`));
    mf.violations.forEach((x) => {
      v.appendChild(el("div", "violation",
        `${esc(x.provider)} · ${esc(x.model || "")} — gap <b>${x.gap_s}s</b> &lt; cooldown <b>${x.cooldown_s}s</b> <span class="tag">${esc(x.call_label || "")} · iter ${x.iteration ?? "?"}</span>`));
    });
    bodyEl.appendChild(v);
  }

  // Per-model stats
  if ((mf.stats || []).length) {
    bodyEl.appendChild(el("div", "phase-label", "Per-model stats"));
    const tbl = el("table", "stats-table");
    tbl.innerHTML = `<thead><tr><th>provider / model</th><th>kind</th><th>✓</th><th>✗</th><th>∅</th><th>total</th><th>avg&nbsp;ms</th><th>top reasons</th></tr></thead>`;
    const tb = el("tbody");
    mf.stats.forEach((s) => {
      const reasons = Object.entries(s.reasons || {}).sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([r, n]) => `${esc(r)}×${n}`).join(", ");
      const tr = el("tr");
      tr.innerHTML = `<td><b>${esc(s.provider)}</b> <span class="mono">${esc(s.model || "")}</span></td>
        <td>${esc(s.kind)}</td><td class="ok">${s.success}</td><td class="err">${s.error}</td>
        <td class="skip">${s.skipped}</td><td>${s.total}</td><td>${s.avg_latency_ms ?? "—"}</td>
        <td class="reasons">${reasons}</td>`;
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    bodyEl.appendChild(tbl);
  }

  // Full ordered timeline (collapsible)
  const d = el("details", "param mflow-timeline");
  d.appendChild(el("summary", null, `full attempt timeline (${(mf.attempts || []).length})`));
  const tl = el("div", "flow-timeline");
  (mf.attempts || []).forEach((a) => {
    const row = renderAttemptRow(a);
    const meta = el("span", "flow-callmeta", `${esc(a.call_label || "")}·i${a.iteration ?? "?"}`);
    row.insertBefore(meta, row.firstChild);
    tl.appendChild(row);
  });
  d.appendChild(tl);
  bodyEl.appendChild(d);

  wrap.appendChild(bodyEl);
  return wrap;
}

// ---------- Init ----------
$("#launch-btn").onclick = launchRun;
$("#drawer-close").onclick = () => showDrawer(false);
$("#overlay").onclick = () => showDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") showDrawer(false); });

fetchRuns();
setInterval(() => { if (!state.es) fetchRuns(); }, 5000);

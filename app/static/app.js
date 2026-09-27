const $ = (id) => document.getElementById(id);
const state = { sid: null, cat: null, busy: false, calls: 0 };

const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

function fillTaskTypes() {
  const role = $("role").value;
  const sel = $("taskType");
  sel.innerHTML = "";
  for (const t of state.cat.catalog.task_types[role] || []) {
    sel.append(new Option(t, t));
  }
  renderRef();
}

function renderRef() {
  const role = $("role").value;
  const cat = state.cat.catalog;
  const wrap = $("ref");
  wrap.innerHTML = "";

  const h = el("h3", null, `P(r) for ${role}`);
  const ul = el("ul");
  const granted = new Set(cat.permissions[role]);
  for (const t of Object.keys(cat.tools).sort()) {
    const li = el("li");
    li.append(el("span", granted.has(t) ? "grant" : "ung", granted.has(t) ? "✓ " : "✕ "));
    li.append(document.createTextNode(t));
    ul.append(li);
  }
  wrap.append(h, ul);

  const scope = cat.scopes[$("taskType").value] || [];
  const h2 = el("h3", null, `S(τ) for ${$("taskType").value}`);
  const ul2 = el("ul");
  for (const t of scope) {
    const li = el("li");
    li.append(el("span", "sc", "· "));
    li.append(document.createTextNode(t));
    ul2.append(li);
  }
  wrap.append(h2, ul2);

  const h3 = el("h3", null, "Drift taxonomy");
  const ul3 = el("ul");
  for (const [k, v] of [
    ["I", "outside the role — C1 catches"],
    ["II", "in role, outside the task — only C2 catches"],
    ["III", "in task, harmful parameters — only detection catches"],
  ]) {
    const li = el("li");
    li.append(el("span", k === "I" ? "I" : k === "II" ? "II" : "III", `[${k}] `));
    li.append(document.createTextNode(v));
    ul3.append(li);
  }
  wrap.append(h3, ul3);
}

function addMsg(who, text, cls) {
  const empty = document.querySelector("#chat .empty");
  if (empty) empty.remove();
  const m = el("div", `msg ${cls}`);
  m.append(el("div", "who", who));
  m.append(el("div", "body", text));
  $("chat").append(m);
  $("chat").scrollTop = $("chat").scrollHeight;
  return m;
}

function tag(text, cls) {
  return el("span", `tag ${cls || ""}`, text);
}

function addDecision(ev, result) {
  const empty = document.querySelector("#log .empty");
  if (empty) empty.remove();
  state.calls += 1;
  $("logCount").textContent = `${state.calls} call${state.calls === 1 ? "" : "s"}`;

  const e = ev.event;
  const card = el("div", `card ${e.decision}`);

  const top = el("div", "top");
  top.append(el("span", "act", e.action));
  top.append(tag(e.decision.toUpperCase(), e.decision === "allow" ? "ok" : "no"));
  if (e.deny_layer) top.append(tag(`denied by ${e.deny_layer}`));
  if (e.drift_type && e.drift_type !== "none") top.append(tag(`type ${e.drift_type}`, e.drift_type));
  if (e.harm) top.append(tag(`HARM ${e.harm_rule_ids.join(", ")}`, "harm"));
  top.append(tag(e.in_role ? "in role" : "OUT OF ROLE", e.in_role ? "" : "no"));
  top.append(tag(e.in_task ? "in task" : "outside task", e.in_task ? "" : "no"));
  if (e.tainted_context) top.append(tag("tainted ctx", "III"));
  card.append(top);

  card.append(el("pre", null, JSON.stringify(e.params)));

  if (e.audit && Object.keys(e.audit).length) {
    const a = el("details");
    a.append(el("summary", null, "audit before-values"));
    a.append(el("pre", null, JSON.stringify(e.audit, null, 2)));
    card.append(a);
  }
  if (e.format_error_detail) {
    card.append(el("div", "tag no", `format error: ${e.format_error_detail}`));
  }
  if (result && result.error) {
    const d = el("details");
    d.append(el("summary", null, "what the agent was told"));
    d.append(el("pre", null, result.error));
    card.append(d);
  } else if (result && result.ok) {
    const d = el("details");
    d.append(el("summary", null, "tool result"));
    d.append(el("pre", null, JSON.stringify(result.data, null, 2)));
    card.append(d);
  }

  $("log").append(card);
  $("log").scrollTop = $("log").scrollHeight;
}

function renderSummary(s) {
  $("sessPill").textContent = `${s.role} · ${s.control} · ${s.task_type} · ${s.model}`;
  const chips = $("chips");
  chips.innerHTML = "";
  const mk = (label, val, bad) => {
    const c = el("span", `chip${bad ? " bad" : ""}`);
    c.append(document.createTextNode(label + " "));
    c.append(el("b", null, String(val)));
    chips.append(c);
  };
  mk("calls", s.calls);
  mk("denied", s.denied, s.denied > 0);
  mk("harmful", s.harmful, s.harmful > 0);
  mk("executed harmful", s.executed_harmful, s.executed_harmful > 0);
  for (const [k, v] of Object.entries(s.drift_counts || {})) {
    if (k !== "none") mk(`type ${k}`, v, true);
  }
}

async function newSession() {
  try {
    const r = await api("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: $("role").value,
        control: $("control").value,
        task_type: $("taskType").value,
        model: $("model").value,
      }),
    });
    state.sid = r.session_id;
    state.calls = 0;
    $("chat").innerHTML = "";
    $("log").innerHTML = "";
    $("log").append(el("div", "empty", "Every tool call the agent makes is decided here, by the PEP, outside the agent."));
    $("logCount").textContent = "0 calls";
    renderSummary(r.summary);
    $("input").disabled = false;
    $("send").disabled = false;
    $("input").focus();
  } catch (e) {
    addMsg("error", String(e.message || e), "err");
  }
}

async function send() {
  const text = $("input").value.trim();
  if (!text || state.busy || !state.sid) return;
  state.busy = true;
  $("send").disabled = true;
  $("input").value = "";
  addMsg("you", text, "user");

  const pending = el("div", "msg sys");
  pending.append(el("div", "body", "thinking…"));
  $("chat").append(pending);

  try {
    const res = await fetch(`/api/session/${state.sid}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let gotText = false;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!frame.startsWith("data: ")) continue;
        const data = frame.slice(6);
        if (data === "[DONE]") continue;
        const ev = JSON.parse(data);
        if (ev.type === "assistant") {
          if (!gotText) { pending.remove(); gotText = true; }
          addMsg(`${state.sid ? $("role").value : "agent"}`, ev.text, "agent");
        } else if (ev.type === "decision") {
          if (!gotText) { pending.remove(); gotText = true; }
          addDecision(ev, ev.result);
        } else if (ev.type === "end") {
          if (!gotText) { pending.remove(); gotText = true; }
          addMsg("system", `turn limit reached (${ev.reason})`, "sys");
        } else if (ev.type === "error") {
          pending.remove();
          gotText = true;
          addMsg("error", ev.error, "err");
        } else if (ev.type === "summary") {
          renderSummary(ev.summary);
        }
      }
    }
    if (!gotText) pending.remove();
  } catch (e) {
    pending.remove();
    addMsg("error", String(e.message || e), "err");
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("input").focus();
  }
}

async function boot() {
  const r = await api("/api/catalog");
  state.cat = r;
  for (const role of r.catalog.roles) $("role").append(new Option(role, role));
  for (const [k, label] of Object.entries(r.models)) $("model").append(new Option(label, k));
  if (!r.has_groq_key) {
    $("fence").textContent = "GROQ_API_KEY not set — the paid model will fail";
  }
  fillTaskTypes();
  $("role").addEventListener("change", fillTaskTypes);
  $("taskType").addEventListener("change", renderRef);
  $("newSession").addEventListener("click", newSession);
  $("send").addEventListener("click", send);
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
}

boot();

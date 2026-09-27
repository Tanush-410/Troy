# Interactive UI and role restructure

Working notes. Nothing here is frozen yet.

## 1. What the paper argues

Agents act inside business systems with real permissions and are genuinely capable of
doing the job. The problem is not capability, it is that intent is not stable. Intent
drifts over long horizons (D1), under underspecified or role-conflicting instructions
(D2), and after the agent reads adversarial content written by someone it does not
control (D3).

RBAC is the standard containment mechanism, and it assumes the intent the policy was
written for is still the intent operating. That assumption is what fails.

Drift is split into three types, and this split is the paper's core contribution:

| Type | Condition | Static RBAC (C1) | Task-scoped (C2) | Needs detection |
|---|---|---|---|---|
| I scope creep | action outside `P_r` | catches | catches | no |
| II out-of-task use | in `P_r`, outside `S_tau` | misses | catches | no |
| III in-role misuse | in `S_tau`, harmful parameters or frequency | misses | misses | **yes** |

So the argument is not "RBAC does not work". It is that RBAC works well on Type I,
partly works on Type II once you scope to the task, and is structurally blind to
Type III. Type III needs behavioural detection, and the only place a detector can
honestly sit is outside the agent, reading the permission log the enforcement point
already writes.

## 2. Status of the freeze

The harness was frozen at commit `9e94387`. **No real-model episode has ever run** —
`logs/` contains only `scripted-C1` and `scripted-C2`. Restructuring the roles now
costs nothing in disclosure terms: change the tables, re-freeze, start the pilot from
a clean frozen state. This is a design decision, not a retrofit.

## 3. Model

`openai/gpt-oss-120b` on Groq. Verified to emit clean tool calls through the existing
provider stack, ~0.6 s per turn.

`qwen/qwen3.8-27b` is **not usable on the current plan tier**: the free tier caps
output at 1000 tokens/min (OTPM) and the harness requests more, so every call returns
429. A pilot needs two models, so this needs either a paid tier or a genuinely local
open-weight model via Ollama. Unresolved.

## 4. The three agents

As stated by the author. This **restructures** the frozen tables: `support` loses
refunds, `listing` and `price_intel` merge, and `billing` is new.

| Role | Grants | Explicitly denied |
|---|---|---|
| support chat | read tickets, read orders, reply to customer, escalate to human | refunds, discounts, listing/catalog edits, data export |
| billing & discount | issue refund, issue discount, update account-level billing fields, escalate to human | listing/catalog edits, data export |
| listing & price intel | read/update product listings, set pricing, read price history, escalate to human | refunds, billing actions, customer data access |

The four tools no role grants (`export_customer_data`, `delete_account`,
`bulk_update_prices`, `send_marketing_email`) stay ungranted, so Type I stays
observable. Every tool schema is still exposed to every agent; enforcement is entirely
at the PEP.

### 4.1 What this costs

- **The cross-agent taint path is lost.** In the frozen design a poisoned competitor
  page produced a bad price report that a *different* agent then acted on, so taint
  crossed an agent boundary. Merging listing and price intel turns that into a
  within-agent loop, which is a weaker claim. This is the strongest drift story in the
  project and it is the main argument against the merge. Keeping `price_intel`
  separate and adding `billing` as a fourth role preserves it.
- **Billing needs tools that do not exist**: some `issue_discount`, some
  `update_billing_fields`. Plus new task types, new harm rules, new success checkers,
  and a new role prompt.
- **Refunds move**, so harm rule S1 and the `process_refund` task type change owner.
  `S_τ` for support loses `issue_refund`.

### 4.2 Open questions

1. Does `support` still get `update_shipping_address`? Unmentioned. Without it the
   `update_address` task type has no owner.
2. "Bring a product / take them to the order screen" — read as *UI affordances* of the
   chat app (show a product card, link to an order page), not as new tool grants.
   Confirm.
3. Is `read_listing` still granted to all three? It is read-only catalog visibility
   and `answer_query` lists it in `S_τ`.
4. Is the merge of listing and price intel really wanted, given 4.1?

## 5. The interactive UI

A normal application: pick an agent, talk to it, watch it work, watch the policy
decide. Baseline is **C1 static RBAC**; C2 is already implemented and will be
selectable.

### 5.1 Why it is not a web framework

No web framework is installed and the dependency set is pinned. A stdlib
`http.server` + Server-Sent Events + one vanilla JS page adds **zero** dependencies,
keeps `uv.lock` frozen, and runs today. SSE gives genuine streaming, which is what
makes the permission decisions feel live as the agent works.

### 5.2 Why the agent loop had to be rewritten

`LLMAgent.run_task` is a single blocking call that loops to completion internally, so a
human cannot interleave. The `Session` protocol
(`add_user` / `step` / `add_tool_results`) *is* resumable, so the UI drives it
directly. All tool calls still go through `gateway.call`, so the PEP stays the only
path to the environment and the only writer of the log.

### 5.3 How live logging works

`PEP` takes any object with `.write(record)` as its event sink — duck typing, no
subclassing needed. The UI passes an in-memory sink that appends to a per-session list
and pushes onto an SSE queue. So the manual session's log appears in the GUI as it
happens, with no tailing of files and no polling.

### 5.4 The demo/data fence

Interactive sessions are **not experimental data**, and the code must enforce it:

- A human typing free-form messages has no seed, so rule 6 (reproduce from config and
  seed) cannot hold for it.
- `pi*_tau`, the reference policy, requires the same agent on the same task type with
  no drift induction. Human input destroys that.
- Rule 1 forbids numbers that are not computed from real runs. A demo transcript is not
  a real run.

So: UI sessions write to their own store, every record tagged
`interactive_ui: true`, and the analysis readers refuse to load them. This is a
`ProvenanceError`, not a convention, for the same reason the detector reads only
`DetectorEvent`.

## 6. Build order

1. `app/session.py` — turn-by-turn session, owns the PEP, in-memory sink.
2. `app/server.py` — JSON API + SSE.
3. `app/static` — chat pane, live policy panel, role/control/task-type pickers.
4. Fence UI sessions out of `analysis/`.
5. Manual end-to-end test against Groq under C1.
6. Injection toggle, so A1 taint drift can be demonstrated on demand.
7. Only then: restructure the roles per section 4, once 4.2 is answered.

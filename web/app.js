"use strict";

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const STATE_NAMES = ["gray", "yellow", "green"];
const FB_CHARS = [".", "y", "g"];

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    const msg = Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) node.append(c);
  return node;
}

function tile(letter, state, interactive, onClick) {
  const cls = `tile ${state === null ? "" : "s" + state} ${letter ? "" : "blank"}`.trim();
  const label = letter ? `${letter.toUpperCase()}, ${state === null ? "no colour" : STATE_NAMES[state]}` : "empty";
  const t = el(interactive ? "button" : "div", { class: cls, "aria-label": label, text: letter || "·" });
  if (interactive) {
    t.type = "button";
    t.addEventListener("click", onClick);
  }
  return t;
}

function rowOf(word, fb, reveal = false) {
  const row = el("div", { class: "row" + (reveal ? " reveal" : "") });
  [...word].forEach((ch, i) => row.append(tile(ch, fb[i], false)));
  return row;
}

const fbToStates = (s) => [...s].map((c) => (c === "g" ? 2 : c === "y" ? 1 : 0));
const statesToFb = (a) => a.map((s) => FB_CHARS[s]).join("");

/* ---------- tabs and modes ---------- */
function selectTab(name) {
  for (const t of ["wordle", "connections"]) {
    $(`panel-${t}`).hidden = t !== name;
    $(`tab-${t}`).classList.toggle("active", t === name);
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
  }
  history.replaceState(null, "", `#${name}`);
}
$("tab-wordle").addEventListener("click", () => selectTab("wordle"));
$("tab-connections").addEventListener("click", () => selectTab("connections"));

function selectMode(mode) {
  $("assist").hidden = mode !== "assist";
  $("watch").hidden = mode !== "watch";
  for (const m of ["assist", "watch"]) {
    $(`mode-${m}`).classList.toggle("active", m === mode);
    $(`mode-${m}`).setAttribute("aria-pressed", String(m === mode));
  }
}
$("mode-assist").addEventListener("click", () => selectMode("assist"));
$("mode-watch").addEventListener("click", () => selectMode("watch"));

/* ---------- assistant ---------- */
const game = { history: [], states: [0, 0, 0, 0, 0], seq: 0 };

function renderBoard() {
  const board = $("board");
  board.replaceChildren(...game.history.map((h, i) => rowOf(h.guess, fbToStates(h.feedback), i === game.history.length - 1 && game.justAdded)));
  game.justAdded = false;
  $("undo").disabled = game.history.length === 0;
}

function renderEntry() {
  const word = $("guess").value.toLowerCase();
  const row = $("entry-tiles");
  row.className = "row";
  row.replaceChildren(
    ...[0, 1, 2, 3, 4].map((i) =>
      tile(word[i] || "", word[i] ? game.states[i] : null, Boolean(word[i]), () => {
        game.states[i] = (game.states[i] + 1) % 3;
        renderEntry();
        row.children[i].focus(); // re-rendering replaces the buttons; keep keyboard focus on this tile
      })
    )
  );
  $("add").disabled = word.length !== 5;
}

// Enter submits from the tiles too (Space or click cycles a tile's colour)
$("entry-tiles").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    addGuess();
  }
});

function showError(id, msg) {
  const e = $(id);
  e.hidden = !msg;
  e.textContent = msg || "";
}

function bitsCell(bits, max) {
  const bar = el("div", { class: "bitbar" }, [
    el("i", { style: `width:${Math.max(4, (bits / max) * 100)}%` }),
    el("span", { text: `${bits.toFixed(2)} bits` }),
  ]);
  return bar;
}

function renderSuggestions(r) {
  $("remaining").textContent = r.remaining.toLocaleString();
  $("bits").textContent = r.bits_remaining.toFixed(1);
  $("message").hidden = !r.message;
  $("message").textContent = r.message || "";

  const body = $("sugg").querySelector("tbody");
  const max = Math.max(1, ...r.suggestions.map((s) => s.entropy));
  body.replaceChildren(
    ...r.suggestions.map((s) => {
      const word = el("span", { class: "word", text: s.word });
      const cell = el("td", {}, [word]);
      if (s.is_candidate) cell.append(el("span", { class: "badge", text: "possible answer" }));
      const tr = el("tr", { class: "pick", tabindex: "0", title: "Use this guess" }, [
        cell,
        el("td", {}, [bitsCell(s.entropy, max)]),
        el("td", { text: s.expected_remaining < 10 ? s.expected_remaining.toFixed(1) : Math.round(s.expected_remaining).toLocaleString() }),
        el("td", { text: s.worst_case.toLocaleString() }),
      ]);
      const choose = () => {
        $("guess").value = s.word;
        game.states = [0, 0, 0, 0, 0];
        renderEntry();
        $("guess").focus();
      };
      tr.addEventListener("click", choose);
      tr.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), choose()));
      return tr;
    })
  );

  $("cands-n").textContent = r.remaining.toLocaleString();
  const shown = r.candidates.map((w) => el("code", { text: w }));
  if (r.candidates_truncated) shown.push(el("span", { class: "hint", text: `…and ${(r.remaining - r.candidates.length).toLocaleString()} more` }));
  $("cands").replaceChildren(...shown);
  $("cands-wrap").hidden = r.remaining === 0;
}

async function refresh(history = game.history) {
  const mine = ++game.seq;
  const r = await api("/api/wordle/suggest", {
    history,
    hard_mode: $("hard").checked,
    solver: $("ranker").value,
    top_k: 10,
  });
  if (mine !== game.seq) return null; // a newer request superseded this one
  renderSuggestions(r);
  return r;
}

async function addGuess() {
  const guess = $("guess").value.toLowerCase();
  if (guess.length !== 5) return;
  const entry = { guess, feedback: statesToFb(game.states) };
  showError("error", "");
  try {
    const r = await refresh([...game.history, entry]);
    if (r === null) return;
    game.history.push(entry);
    game.justAdded = true;
    $("guess").value = "";
    game.states = [0, 0, 0, 0, 0];
    renderBoard();
    renderEntry();
    if (entry.feedback === "ggggg") {
      $("message").hidden = false;
      $("message").textContent = `Solved in ${game.history.length} guess${game.history.length === 1 ? "" : "es"}.`;
      $("sugg").querySelector("tbody").replaceChildren();
    }
  } catch (e) {
    showError("error", e.message);
  }
}

$("guess").addEventListener("input", (e) => {
  e.target.value = e.target.value.replace(/[^a-zA-Z]/g, "").slice(0, 5);
  renderEntry();
  showError("error", "");
});
$("guess").addEventListener("keydown", (e) => e.key === "Enter" && addGuess());
$("add").addEventListener("click", addGuess);
$("undo").addEventListener("click", async () => {
  game.history.pop();
  renderBoard();
  showError("error", "");
  await refresh().catch((e) => showError("error", e.message));
});
$("reset").addEventListener("click", async () => {
  game.history = [];
  game.states = [0, 0, 0, 0, 0];
  $("guess").value = "";
  renderBoard();
  renderEntry();
  showError("error", "");
  await refresh().catch((e) => showError("error", e.message));
});
$("hard").addEventListener("change", () => refresh().catch((e) => showError("error", e.message)));
$("ranker").addEventListener("change", () => refresh().catch((e) => showError("error", e.message)));

/* ---------- watch the solver ---------- */
let simTimer = null;

async function runSimulation() {
  clearTimeout(simTimer);
  showError("watch-error", "");
  $("sim-result").hidden = true;
  $("sim-board").replaceChildren();
  $("trace").querySelector("tbody").replaceChildren();
  const secret = $("secret").value.trim().toLowerCase();
  $("run").disabled = true;
  try {
    const r = await api("/api/wordle/simulate", {
      answer: secret || null,
      solver: $("solver").value,
      hard_mode: $("hard").checked,
    });
    let i = 0;
    const step = () => {
      const s = r.steps[i];
      $("sim-board").append(rowOf(s.guess, fbToStates(s.feedback), true));
      $("trace").querySelector("tbody").append(
        el("tr", {}, [
          el("td", { text: String(i + 1) }),
          el("td", {}, [el("span", { class: "word", text: s.guess })]),
          el("td", { text: `${s.remaining_before.toLocaleString()} → ${s.remaining_after.toLocaleString()}` }),
          el("td", { text: `${s.expected_bits.toFixed(2)} bits` }),
          el("td", { text: `${s.actual_bits.toFixed(2)} bits` }),
        ])
      );
      i += 1;
      if (i < r.steps.length) simTimer = setTimeout(step, 650);
      else {
        $("sim-result").hidden = false;
        $("sim-result").textContent = r.solved
          ? `Solved ${r.answer.toUpperCase()} in ${r.guesses} guess${r.guesses === 1 ? "" : "es"}.`
          : `Did not solve ${r.answer.toUpperCase()} within ${r.guesses} guesses.`;
        $("run").disabled = false;
      }
    };
    step();
  } catch (e) {
    showError("watch-error", e.message);
    $("run").disabled = false;
  }
}
$("run").addEventListener("click", runSimulation);
$("secret").addEventListener("keydown", (e) => e.key === "Enter" && runSimulation());
$("secret").addEventListener("input", (e) => (e.target.value = e.target.value.replace(/[^a-zA-Z]/g, "").slice(0, 5)));

/* ---------- boot ---------- */
selectTab(location.hash === "#connections" ? "connections" : "wordle");
renderBoard();
renderEntry();
refresh().catch((e) => showError("error", `Could not reach the solver: ${e.message}`));

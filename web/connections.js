"use strict";

/* Connections tab. Relies on helpers ($, el, api) defined in app.js. */
const conn = { solved: [], wrong: [], pendingHint: null };

function buildGrid() {
  const grid = $("c-grid");
  grid.replaceChildren(
    ...Array.from({ length: 16 }, (_, i) => {
      const input = el("input", { type: "text", "aria-label": `Word ${i + 1}`, autocomplete: "off", spellcheck: "false", maxlength: "40" });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          const next = grid.children[i + 1];
          if (next) next.focus();
          else $("c-solve").focus();
        }
      });
      input.addEventListener("input", () => resetResults());
      return input;
    })
  );
}

const words = () => [...$("c-grid").children].map((i) => i.value.trim());

function setStatus(msg) {
  $("c-status").hidden = !msg;
  $("c-status").textContent = msg || "";
}

function resetResults() {
  conn.solved = [];
  conn.wrong = [];
  conn.pendingHint = null;
  for (const id of ["c-solution", "c-hintbox", "c-solved-wrap"]) $(id).hidden = true;
  $("c-empty").hidden = false;
  for (const i of $("c-grid").children) i.classList.remove("dim");
}

function groupCard(g, cls) {
  const pct = Math.round(g.confidence * 100);
  return el("div", { class: `c-group ${cls}` }, [
    el("div", { class: "c-words", text: g.words.join(" · ") }),
    g.solved
      ? el("div", { class: "c-conf", text: "Marked solved" })
      : el("div", { class: "c-conf" }, [
          el("span", { text: `${pct}% sure` }),
          el("span", { class: "meter" }, [el("i", { style: `width:${Math.max(3, pct)}%` })]),
        ]),
  ]);
}

function markSolvedWords() {
  const solvedWords = new Set(conn.solved.flat().map((w) => w.toLowerCase()));
  for (const i of $("c-grid").children) i.classList.toggle("dim", solvedWords.has(i.value.trim().toLowerCase()));
}

async function solveAll() {
  showError("c-error", "");
  setStatus("Solving...");
  $("c-solve").disabled = true;
  try {
    const r = await api("/api/connections/solve", { words: words() });
    conn.solved = [];
    conn.wrong = [];
    $("c-hintbox").hidden = true;
    $("c-solved-wrap").hidden = true;
    $("c-empty").hidden = true;
    $("c-solution").hidden = false;
    $("c-expected").textContent = `Expected correct groups: about ${r.expected_correct.toFixed(1)} of 4. Most confident first.`;
    $("c-groups").replaceChildren(...r.groups.map((g, i) => groupCard(g, `g${i}`)));
    markSolvedWords();
  } catch (e) {
    showError("c-error", e.message);
  } finally {
    setStatus("");
    $("c-solve").disabled = false;
  }
}

function renderSolved() {
  $("c-solved-wrap").hidden = conn.solved.length === 0;
  $("c-solved").replaceChildren(...conn.solved.map((w) => groupCard({ words: w, solved: true, confidence: 1 }, "solved")));
  markSolvedWords();
}

async function giveHint() {
  showError("c-error", "");
  setStatus("Thinking...");
  $("c-hint").disabled = true;
  try {
    const r = await api("/api/connections/hint", { words: words(), solved: conn.solved, wrong: conn.wrong });
    $("c-empty").hidden = true;
    $("c-solution").hidden = true;
    if (!r.group) {
      $("c-hintbox").hidden = true;
      setStatus(r.message);
      return;
    }
    conn.pendingHint = r.group.words;
    $("c-hintbox").hidden = false;
    $("c-hintgroup").replaceChildren(groupCard(r.group, "g0"));
    renderSolved();
  } catch (e) {
    showError("c-error", e.message);
  } finally {
    if ($("c-status").textContent === "Thinking...") setStatus("");
    $("c-hint").disabled = false;
  }
}

$("c-solve").addEventListener("click", solveAll);
$("c-hint").addEventListener("click", giveHint);
$("c-ok").addEventListener("click", () => {
  if (!conn.pendingHint) return;
  conn.solved.push(conn.pendingHint);
  conn.pendingHint = null;
  $("c-hintbox").hidden = true;
  renderSolved();
  if (conn.solved.length < 4) giveHint();
  else setStatus("All four groups found.");
});
$("c-no").addEventListener("click", () => {
  if (!conn.pendingHint) return;
  conn.wrong.push(conn.pendingHint);
  conn.pendingHint = null;
  giveHint();
});
$("c-clear").addEventListener("click", () => {
  for (const i of $("c-grid").children) i.value = "";
  resetResults();
  showError("c-error", "");
  setStatus("");
  $("c-grid").children[0].focus();
});
$("c-fill").addEventListener("click", () => {
  const parts = $("c-paste").value.split(/[\n,;\t]+/).map((s) => s.trim()).filter(Boolean);
  if (parts.length !== 16) {
    showError("c-error", `Found ${parts.length} words in the pasted text; need exactly 16.`);
    return;
  }
  showError("c-error", "");
  [...$("c-grid").children].forEach((input, i) => (input.value = parts[i]));
  resetResults();
});

buildGrid();

fetch("/api/connections/info")
  .then((r) => r.json())
  .then((info) => {
    $("c-unavailable").hidden = info.available;
    $("c-main").hidden = !info.available;
  })
  .catch(() => {});

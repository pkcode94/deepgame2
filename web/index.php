<?php
// Einfache Web?UI zum Abfragen des MultiAgentFractalCore-Server (lokal laufender C#-Server).
// Erwartete Server-Endpunkte (JSON):
// POST /api/query  -> { "question": "...", "context": ["..."] }  => { "id": "...", "answer": "...", "confidence": 0.9, "context": [...] }
// GET  /api/history -> [ { "id": "...", "question": "...", "answer": "...", "timestamp": "..." }, ... ]
// UI speichert lokale History in localStorage; Server?History optional.

?>
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>DeepSeekX — Fractal Agent (User)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 20px; max-width: 900px; }
    textarea { width: 100%; height: 80px; }
    .btn { padding: 8px 12px; margin: 6px 0; }
    .panel { border: 1px solid #ddd; padding: 10px; margin-top:12px; border-radius:6px; }
    .history-item { padding:6px; border-bottom:1px solid #eee; }
  </style>
</head>
<body>
  <h1>DeepSeekX — Fractal Agent</h1>

  <div class="panel">
    <label for="question">Frage / Eingabe:</label>
    <textarea id="question" placeholder="Stelle deine Frage oder gib ein kurzes Kontext-Textfragment ein..."></textarea>

    <label for="context">Optionaler Kontext (eine Zeile pro Eintrag):</label>
    <textarea id="context" placeholder="Vorherige Antworten, Kontext, ... (optional)"></textarea>

    <div>
      <button id="askBtn" class="btn">Frage senden</button>
      <button id="clearBtn" class="btn">Eingaben löschen</button>
    </div>

    <div id="result" class="panel" style="display:none;">
      <strong>Antwort:</strong>
      <div id="answerText" style="white-space:pre-wrap;margin-top:8px;"></div>
      <div style="margin-top:8px;">
        <button id="saveLocal" class="btn">In lokale History speichern</button>
      </div>
    </div>
  </div>

  <h2>Lokale History</h2>
  <div id="history" class="panel"></div>

<script>
const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');
const questionEl = document.getElementById('question');
const contextEl = document.getElementById('context');
const resultPanel = document.getElementById('result');
const answerText = document.getElementById('answerText');
const saveLocal = document.getElementById('saveLocal');
const historyEl = document.getElementById('history');

const API_BASE = '/api'; // Erwarteter Pfad zum lokal laufenden C#-Server

function renderHistory() {
  const hist = JSON.parse(localStorage.getItem('fractal_history') || '[]');
  if (!hist.length) {
    historyEl.innerHTML = '<div>Keine Einträge.</div>';
    return;
  }
  historyEl.innerHTML = hist.map(h => `
    <div class="history-item">
      <div><strong>Frage:</strong> ${escapeHtml(h.question)}</div>
      <div><strong>Antwort:</strong> ${escapeHtml(h.answer)}</div>
      <div style="font-size:0.85em;color:#666">${new Date(h.timestamp).toLocaleString()}</div>
    </div>`).join('');
}

function escapeHtml(s){ return (s+'').replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }

askBtn.addEventListener('click', async () => {
  const q = questionEl.value.trim();
  if (!q) { alert('Bitte Frage eingeben.'); return; }
  const ctx = contextEl.value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  askBtn.disabled = true;
  askBtn.textContent = 'Sende...';

  try {
    const resp = await fetch(API_BASE + '/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, context: ctx })
    });
    if (!resp.ok) throw new Error('Serverfehler ' + resp.status);
    const data = await resp.json();
    answerText.textContent = data.answer ?? '(keine Antwort)';
    resultPanel.style.display = 'block';
    // optionally push into local history automatically
    const hist = JSON.parse(localStorage.getItem('fractal_history') || '[]');
    hist.unshift({ id: data.id||null, question: q, answer: data.answer||'', timestamp: Date.now() });
    localStorage.setItem('fractal_history', JSON.stringify(hist.slice(0,200)));
    renderHistory();
  } catch (err) {
    alert('Fehler: ' + err.message);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Frage senden';
  }
});

clearBtn.addEventListener('click', () => { questionEl.value=''; contextEl.value=''; resultPanel.style.display='none'; });

saveLocal.addEventListener('click', () => {
  const hist = JSON.parse(localStorage.getItem('fractal_history') || '[]');
  const q = questionEl.value.trim();
  hist.unshift({ id: null, question: q, answer: answerText.textContent, timestamp: Date.now() });
  localStorage.setItem('fractal_history', JSON.stringify(hist.slice(0,200)));
  renderHistory();
});

renderHistory();
</script>
</body>
</html>

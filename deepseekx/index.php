<?php
// Modified to ensure detokenization is serverside — index.php displays server answer string directly and trims at <eos>.
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
    input[type=text] { width:100%; padding:6px; }
    .small { font-size:0.9em; color:#666 }
  </style>
</head>
<body>
  <h1>DeepSeekX — Fractal Agent</h1>

  <div class="panel">
    <label for="apiBase">API Basis-URL (fest auf C#-Server):</label>
    <input type="text" id="apiBase" value="http://localhost:5000/api" disabled />
    <div class="small">Die UI verwendet fest <code>http://localhost:5000/api</code>. Wenn dein C#-Server auf anderem Port läuft, passe diesen Wert in der Datei `index.php` an.</div>

    <hr />

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
        <button id="suggestBtn" class="btn">Vorschlag einreichen</button>
      </div>

      <div id="suggestBox" style="display:none;margin-top:8px;">
        <label for="suggestion">Vorschlag für Antwort:</label>
        <textarea id="suggestion" placeholder="Formuliere einen alternativen Antwortvorschlag..."></textarea>
        <div>
          <button id="sendSuggest" class="btn">Vorschlag senden</button>
          <button id="cancelSuggest" class="btn">Abbrechen</button>
        </div>
        <div id="suggestStatus" class="small"></div>
      </div>
    </div>
  </div>

  <h2>Lokale History (Session)</h2>
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
const apiBaseEl = document.getElementById('apiBase');
const suggestBtn = document.getElementById('suggestBtn');
const suggestBox = document.getElementById('suggestBox');
const suggestionEl = document.getElementById('suggestion');
const sendSuggest = document.getElementById('sendSuggest');
const cancelSuggest = document.getElementById('cancelSuggest');
const suggestStatus = document.getElementById('suggestStatus');

// Hardcoded API base URL matching the C# HttpListener prefix
function getApiBase() { return apiBaseEl.value.trim() || '/api'; }

// Session id for grouping context/history; persistent in localStorage
function getSessionId() {
  let sid = localStorage.getItem('fractal_session_id');
  if (!sid) {
    sid = 'sess-' + Math.random().toString(36).slice(2,12);
    localStorage.setItem('fractal_session_id', sid);
  }
  return sid;
}
const SESSION_ID = getSessionId();

function historyKey() { return 'fractal_history_' + SESSION_ID; }

function renderHistory() {
  const hist = JSON.parse(localStorage.getItem(historyKey()) || '[]');
  if (!hist.length) {
    historyEl.innerHTML = '<div>Keine Einträge.</div>';
    return;
  }
  historyEl.innerHTML = hist.map(h => `
    <div class="history-item">
      <div><strong>Frage:</strong> ${escapeHtml(h.question)}</div>
      <div><strong>Antwort:</strong> ${escapeHtml(h.answer)}</div>
      <div class="small">${new Date(h.timestamp).toLocaleString()}</div>
    </div>`).join('');
}

function escapeHtml(s){ return (s+'').replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }

let lastInteractionId = null;

askBtn.addEventListener('click', async () => {
  const q = questionEl.value.trim();
  if (!q) { alert('Bitte Frage eingeben.'); return; }
  const ctx = contextEl.value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  askBtn.disabled = true;
  askBtn.textContent = 'Sende...';

  const apiBase = getApiBase();
  const url = apiBase.replace(/\/$/, '') + '/query';

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, context: ctx })
    });
    if (!resp.ok) {
      let bodyText = '';
      try { bodyText = await resp.text(); } catch { }
      throw new Error('Serverfehler ' + resp.status + ' beim Aufruf ' + url + ' — ' + bodyText);
    }
    const data = await resp.json();
    // Trim answer at first <eos> (case-insensitive)
    let raw = data.answer ?? '';
    const lower = raw.toLowerCase();
    const eosIdx = lower.indexOf('<eos>');
    let display = eosIdx >= 0 ? raw.substring(0, eosIdx).trim() : raw;
    if (!display) display = '(keine Antwort)';

    answerText.textContent = display;
    resultPanel.style.display = 'block';
    lastInteractionId = data.id || null;

    // store in per-session history (store trimmed display)
    const hist = JSON.parse(localStorage.getItem(historyKey()) || '[]');
    hist.unshift({ id: data.id||null, question: q, answer: display, timestamp: Date.now() });
    localStorage.setItem(historyKey(), JSON.stringify(hist.slice(0,200)));
    renderHistory();
  } catch (err) {
    alert('Fehler: ' + err.message + '\n\nHinweis: Überprüfe die API Basis-URL und dass der C#-Server läuft und CORS erlaubt ist.');
    console.error(err);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Frage senden';
  }
});

clearBtn.addEventListener('click', () => { questionEl.value=''; contextEl.value=''; resultPanel.style.display='none'; });

saveLocal.addEventListener('click', () => {
  const hist = JSON.parse(localStorage.getItem(historyKey()) || '[]');
  const q = questionEl.value.trim();
  hist.unshift({ id: lastInteractionId, question: q, answer: answerText.textContent, timestamp: Date.now() });
  localStorage.setItem(historyKey(), JSON.stringify(hist.slice(0,200)));
  renderHistory();
});

suggestBtn.addEventListener('click', () => {
  suggestBox.style.display = 'block';
  suggestionEl.focus();
});

cancelSuggest.addEventListener('click', () => {
  suggestBox.style.display = 'none';
  suggestionEl.value = '';
  suggestStatus.textContent = '';
});

sendSuggest.addEventListener('click', async () => {
  const text = suggestionEl.value.trim();
  if (!text) { alert('Bitte Vorschlag eingeben.'); return; }
  if (!lastInteractionId) { alert('Keine zugeordnete Interaktion vorhanden. Bitte zuerst eine Frage senden.'); return; }

  sendSuggest.disabled = true;
  sendSuggest.textContent = 'Sende...';

  try {
    const resp = await fetch(getApiBase().replace(/\/$/, '') + '/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ interactionId: lastInteractionId, suggestion: text, sessionId: SESSION_ID })
    });
    if (!resp.ok) {
      let bodyText = '';
      try { bodyText = await resp.text(); } catch { }
      throw new Error('Serverfehler ' + resp.status + ' — ' + bodyText);
    }
    const data = await resp.json();
    suggestStatus.textContent = 'Vorschlag gesendet (id=' + (data.id||'') + '). Vielen Dank.';
    suggestionEl.value = '';
  } catch (err) {
    suggestStatus.textContent = 'Fehler beim Senden: ' + err.message;
    console.error(err);
  } finally {
    sendSuggest.disabled = false;
    sendSuggest.textContent = 'Vorschlag senden';
  }
});

// initial render
renderHistory();
</script>
</body>
</html>

<?php
// Admin UI zum Reinforcement: Erwartete Server-Endpunkte:
// GET  /api/interactions         -> [ { "id":"...", "question":"...", "answer":"...", "context":[...], "timestamp":"..." }, ... ]
// POST /api/reinforce           -> { "id":"...", "positive": true, "note":"optional", "examples": [ { "question":"", "answer":"" }, ... ] }
// POST /api/retrain_window      -> { "examples": [ { "question":"", "answer":"", "label": 1 }, ... ] }
// POST /api/suggest             -> { "interactionId":"...", "suggestion":"...", "sessionId":"..." } => { id, status }
// POST /api/suggestions/verify  -> { "id":"...", "accept": true } => { id, status }
// Diese Admin-Seite ruft Interaktionen ab und erlaubt das Verstärken von Antworten (trainiere das CombinatorialPathGate auf einem Fenster vorheriger Antworten).

?>
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>DeepSeekX — Admin</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:20px;max-width:1000px}
    .item{border:1px solid #ddd;padding:10px;margin-bottom:10px;border-radius:6px}
    button{padding:6px 10px;margin-right:6px}
    textarea{width:100%;height:60px}
    input[type=text]{padding:6px;width:420px}
    .small{font-size:0.9em;color:#666}
  </style>
</head>
<body>
  <h1>Admin — Reinforcement</h1>
  <p>Diese Seite zeigt Server-Interaktionen und erlaubt das Verstärken von Antworten (trainiere das CombinatorialPathGate auf einem Fenster vorheriger Antworten).</p>

  <div style="margin-bottom:12px;">
    API Basis-URL: <input id="apiBase" type="text" placeholder="http://localhost:5000/api" />
    <button id="saveApi" class="btn">Set</button>
  </div>

  <div>
    <button id="refresh">Interaktionen neu laden</button>
    <button id="trainWindow">Trainiere aktuelles Fenster (serverseitig)</button>
  </div>

  <div id="list"></div>

<script>
// Determine API base; default to localhost:5000/api
function getApiBase(){
  const stored = localStorage.getItem('admin_api_base');
  if (stored && stored.trim().length>0) return stored.replace(/\/$/, '');
  const el = document.getElementById('apiBase');
  if (el && el.value && el.value.trim().length>0) return el.value.trim().replace(/\/$/, '');
  return 'http://localhost:5000/api';
}

// initialize input from storage
(function(){ const el = document.getElementById('apiBase'); const v = localStorage.getItem('admin_api_base') || 'http://localhost:5000/api'; el.value = v; })();
document.getElementById('saveApi').addEventListener('click', ()=>{ const v = document.getElementById('apiBase').value.trim(); localStorage.setItem('admin_api_base', v); alert('API Basis-URL gesetzt: ' + v); });

// Normalize interaction object from server (handle PascalCase or camelCase)
function normalizeInteraction(it){
  return {
    id: it.Id || it.id || it.interactionId || '',
    question: it.Question || it.question || it.QuestionText || '',
    answer: it.Answer || it.answer || it.Response || '',
    context: it.Context || it.context || [],
    timestamp: (it.Timestamp || it.timestamp || it.time || 0)
  };
}

async function loadInteractions(){
  const el = document.getElementById('list');
  el.innerHTML = 'Lade...';
  try {
    const url = getApiBase() + '/interactions';
    const r = await fetch(url);
    if(!r.ok){ const body = await r.text(); throw new Error('Server ' + r.status + ' from ' + url + '\n' + body); }
    const items = await r.json();
    if(!items || items.length === 0){ el.innerHTML = '<div>Keine Interaktionen.</div>'; return; }

    // normalize and render
    const normalized = items.map(normalizeInteraction);
    el.innerHTML = normalized.map(i => renderItem(i)).join('');
    attachButtons();
  } catch(e) {
    el.innerHTML = 'Fehler: ' + e.message;
    console.error(e);
    alert('Fehler beim Laden der Interaktionen:\n' + e.message);
  }
}

function renderItem(i){
  const ctx = (i.context || []).join(' | ');
  // ensure valid timestamp
  let tsText = '';
  try{ const t = Number(i.timestamp) || 0; tsText = t>0 ? new Date(t).toLocaleString() : '—'; } catch { tsText = '—'; }
  return `<div class="item" data-id="${i.id}">
    <div><strong>Frage:</strong> ${escapeHtml(i.question)}</div>
    <div><strong>Antwort:</strong> ${escapeHtml(i.answer)}</div>
    <div><strong>Kontext:</strong> ${escapeHtml(ctx)}</div>
    <div style="font-size:0.85em;color:#666">${tsText}</div>
    <div style="margin-top:8px;">
      <button class="btn-pos">+ Verstärken</button>
      <button class="btn-neg">- Verstärken</button>
      <button class="btn-show">Zeige Fenster</button>
    </div>
    <div style="margin-top:8px;">
      <label><strong>Alternativ-Antwort vorschlagen:</strong></label>
      <textarea class="alt-response" placeholder="Gebe hier die alternative Antwort ein..."></textarea>
      <div style="margin-top:6px;">
        <button class="btn-propose">Vorschlag senden</button>
        <button class="btn-accept-reinforce">Annehmen & Verstärken</button>
      </div>
      <div class="alt-status small"></div>
    </div>
    <div class="window" style="display:none;margin-top:8px;">
      <textarea class="window-data" placeholder='JSON-Array von Beispielen: [{"question":"...","answer":"...","label":1}, ...]'></textarea>
      <div><button class="btn-train-window">Trainiere Fenster</button></div>
    </div>
  </div>`;
}

function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }

function attachButtons(){
  document.querySelectorAll('.btn-pos').forEach(b=>b.onclick = async (ev)=>{ const id = ev.target.closest('.item').dataset.id; await sendReinforce(id, true); });
  document.querySelectorAll('.btn-neg').forEach(b=>b.onclick = async (ev)=>{ const id = ev.target.closest('.item').dataset.id; await sendReinforce(id, false); });
  document.querySelectorAll('.btn-show').forEach(b=>b.onclick = (ev)=>{ const node = ev.target.closest('.item').querySelector('.window'); node.style.display = node.style.display === 'none' ? 'block' : 'none'; });
  document.querySelectorAll('.btn-train-window').forEach(b=>b.onclick = async (ev)=>{ const container = ev.target.closest('.item'); const ta = container.querySelector('.window-data'); try { const examples = JSON.parse(ta.value); const url = getApiBase() + '/retrain_window'; const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ examples }) }); if(!r.ok){ const body = await r.text(); throw new Error('HTTP ' + r.status + ' from ' + url + '\n' + body); } alert('Trainingsauftrag gesendet.'); } catch(e){ alert('Ungültiges JSON oder Serverfehler: ' + e.message); } });

  // alt response handlers
  document.querySelectorAll('.btn-propose').forEach(b=>b.onclick = async (ev)=>{
    const container = ev.target.closest('.item');
    const id = container.dataset.id;
    const ta = container.querySelector('.alt-response');
    const statusEl = container.querySelector('.alt-status');
    const text = ta.value.trim();
    if (!text) { alert('Bitte eine alternative Antwort eingeben.'); return; }
    try {
      statusEl.textContent = 'Sende Vorschlag...';
      const url = getApiBase() + '/suggest';
      const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ interactionId: id, suggestion: text, sessionId: 'admin' }) });
      if(!r.ok){ const body = await r.text(); throw new Error('HTTP ' + r.status + ' from ' + url + '\n' + body); }
      const res = await r.json();
      statusEl.textContent = 'Vorschlag gesendet (id=' + (res.id||'') + ')';
    } catch(e){ console.error(e); statusEl.textContent = 'Fehler beim Senden: ' + e.message; alert('Fehler beim Senden des Vorschlags:\n' + e.message); }
  });

  document.querySelectorAll('.btn-accept-reinforce').forEach(b=>b.onclick = async (ev)=>{
    const container = ev.target.closest('.item');
    const id = container.dataset.id;
    const ta = container.querySelector('.alt-response');
    const statusEl = container.querySelector('.alt-status');
    const text = ta.value.trim();
    if (!text) { alert('Bitte eine alternative Antwort eingeben.'); return; }

    if (!confirm('Annehmen und verstärken?')) return;

    try {
      statusEl.textContent = 'Sende Vorschlag und akzeptiere...';
      // 1) create suggestion
      const suggestUrl = getApiBase() + '/suggest';
      const r1 = await fetch(suggestUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ interactionId: id, suggestion: text, sessionId: 'admin' }) });
      if(!r1.ok){ const body = await r1.text(); throw new Error('HTTP ' + r1.status + ' from ' + suggestUrl + '\n' + body); }
      const sres = await r1.json();
      const sugId = sres.id;

      // 2) accept via suggestions verify
      const verifyUrl = getApiBase() + '/suggestions/verify';
      const r2 = await fetch(verifyUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: sugId, accept: true }) });
      if(!r2.ok){ const body = await r2.text(); throw new Error('HTTP ' + r2.status + ' from ' + verifyUrl + '\n' + body); }
      const vres = await r2.json();

      // 3) reinforce the interaction positively
      const reinforceUrl = getApiBase() + '/reinforce';
      const r3 = await fetch(reinforceUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: id, positive: true, note: 'Accepted alt suggestion: ' + sugId }) });
      if(!r3.ok){ const body = await r3.text(); throw new Error('HTTP ' + r3.status + ' from ' + reinforceUrl + '\n' + body); }

      statusEl.textContent = 'Vorschlag akzeptiert und verstärkt.';
      // optionally reload interactions to show updated answer
      await loadInteractions();
    } catch(e){ console.error(e); statusEl.textContent = 'Fehler: ' + e.message; alert('Fehler beim Akzeptieren/Verstärken:\n' + e.message); }
  });
}

async function sendReinforce(id, positive){
  try {
    const url = getApiBase() + '/reinforce';
    const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id, positive }) });
    if(!r.ok){ const body = await r.text(); throw new Error('Server ' + r.status + ' from ' + url + '\n' + body); }
    alert('Reinforcement gesendet.');
    await loadInteractions();
  } catch(e){ alert('Fehler: ' + e.message); console.error(e); }
}

document.getElementById('refresh').addEventListener('click', loadInteractions);
document.getElementById('trainWindow').addEventListener('click', async ()=>{
  try{
    const url = getApiBase() + '/train_recent_window';
    const r = await fetch(url, { method:'POST' });
    if(!r.ok){ const body = await r.text(); throw new Error('Server ' + r.status + ' from ' + url + '\n' + body); }
    alert('Trainingsstart angefordert.');
  } catch(e){ alert('Fehler beim Anfordern: ' + e.message); }
});

// initial
loadInteractions();
</script>
</body>
</html>

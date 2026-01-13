<?php
// Admin UI zum Reinforcement: Erwartete Server-Endpunkte:
// GET  /api/interactions         -> [ { "id":"...", "question":"...", "answer":"...", "context":[...], "timestamp":"..." }, ... ]
// POST /api/reinforce           -> { "id":"...", "positive": true, "note":"optional", "examples": [ { "question":"", "answer":"" }, ... ] }
// POST /api/retrain_window      -> { "examples": [ { "question":"", "answer":"", "label": 1 }, ... ] }
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
  </style>
</head>
<body>
  <h1>Admin — Reinforcement</h1>
  <p>Diese Seite zeigt Server-Interaktionen und erlaubt das Verstärken von Antworten (trainiere das CombinatorialPathGate auf einem Fenster vorheriger Antworten).</p>

  <div>
    <button id="refresh">Interaktionen neu laden</button>
    <button id="trainWindow">Trainiere aktuelles Fenster (serverseitig)</button>
  </div>

  <div id="list"></div>

<script>
const API = '/api';

async function loadInteractions(){
  const el = document.getElementById('list');
  el.innerHTML = 'Lade...';
  try {
    const r = await fetch(API + '/interactions');
    if(!r.ok) throw new Error('Server ' + r.status);
    const items = await r.json();
    if(!items.length){ el.innerHTML = '<div>Keine Interaktionen.</div>'; return; }
    el.innerHTML = items.map(i => renderItem(i)).join('');
    attachButtons();
  } catch(e){
    el.innerHTML = 'Fehler: ' + e.message;
  }
}

function renderItem(i){
  const ctx = (i.context || []).join(' | ');
  return `<div class="item" data-id="${i.id}">
    <div><strong>Frage:</strong> ${escapeHtml(i.question)}</div>
    <div><strong>Antwort:</strong> ${escapeHtml(i.answer)}</div>
    <div><strong>Kontext:</strong> ${escapeHtml(ctx)}</div>
    <div style="font-size:0.85em;color:#666">${new Date(i.timestamp).toLocaleString()}</div>
    <div style="margin-top:8px;">
      <button class="btn-pos">+ Verstärken</button>
      <button class="btn-neg">- Verstärken</button>
      <button class="btn-show">Zeige Fenster</button>
    </div>
    <div class="window" style="display:none;margin-top:8px;">
      <textarea class="window-data" placeholder='JSON-Array von Beispielen: [{"question":"...","answer":"...","label":1}, ...]'></textarea>
      <div><button class="btn-train-window">Trainiere Fenster</button></div>
    </div>
  </div>`;
}

function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }

function attachButtons(){
  document.querySelectorAll('.btn-pos').forEach(b=>b.onclick = async (ev)=>{
    const id = ev.target.closest('.item').dataset.id;
    await sendReinforce(id, true);
  });
  document.querySelectorAll('.btn-neg').forEach(b=>b.onclick = async (ev)=>{
    const id = ev.target.closest('.item').dataset.id;
    await sendReinforce(id, false);
  });
  document.querySelectorAll('.btn-show').forEach(b=>b.onclick = (ev)=>{
    const node = ev.target.closest('.item').querySelector('.window');
    node.style.display = node.style.display === 'none' ? 'block' : 'none';
  });
  document.querySelectorAll('.btn-train-window').forEach(b=>b.onclick = async (ev)=>{
    const container = ev.target.closest('.item');
    const ta = container.querySelector('.window-data');
    try {
      const examples = JSON.parse(ta.value);
      await fetch(API + '/retrain_window', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ examples }) });
      alert('Trainingsauftrag gesendet.');
    } catch(e){ alert('Ungültiges JSON: ' + e.message); }
  });
}

async function sendReinforce(id, positive){
  try {
    const r = await fetch(API + '/reinforce', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id, positive }) });
    if(!r.ok) throw new Error('Server ' + r.status);
    alert('Reinforcement gesendet.');
  } catch(e){ alert('Fehler: ' + e.message); }
}

document.getElementById('refresh').addEventListener('click', loadInteractions);
document.getElementById('trainWindow').addEventListener('click', async ()=>{
  // Optionally triggers server to train on recent window (server decides window size)
  const r = await fetch(API + '/train_recent_window', { method:'POST' });
  if(r.ok) alert('Trainingsstart angefordert.');
  else alert('Fehler beim Anfordern.');
});

loadInteractions();
</script>
</body>
</html>

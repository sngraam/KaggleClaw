/**
 * KaggleClaw — Frontend SSE consumer and event renderer (v2)
 * ES6 vanilla JS, no frameworks
 *
 * Harmony token channels (from harmony docs):
 *   analysis → thinking (chain-of-thought)
 *   final    → model final text output
 *   commentary/call → tool calls
 */

// ── Globals ─────────────────────────────────────────────────────
let eventSource = null;
let stats = { turns: 0, tools: 0, events: 0 };
let isAgentRunning = false;

// Active streaming cards — finalized when a new block starts
let activeTextCard = null;   // "final" channel streaming card
let activeThinkCard = null;  // "analysis" channel streaming card

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Configure marked.js for proper rendering
  marked.setOptions({
    breaks: true,
    gfm: true,
    tables: true,
  });

  loadCompetitionInfo();
  loadHealth();
  connectSSE();
  hljs.highlightAll();
  loadFileTree();
});

// ── SSE Connection ───────────────────────────────────────────────
function connectSSE() {
  if (eventSource) { eventSource.close(); }

  eventSource = new EventSource('/stream');

  eventSource.onopen = () => {
    console.log('[KaggleClaw] SSE connected');
  };

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleEvent(data);
    } catch (err) {
      console.warn('SSE parse error:', err);
    }
  };

  eventSource.onerror = () => {
    console.warn('[KaggleClaw] SSE connection lost, retrying in 3s...');
    setTimeout(connectSSE, 3000);
  };
}

// ── Event Router ────────────────────────────────────────────────
function handleEvent(data) {
  if (data.type === 'ping') return;

  stats.events++;
  updateStats();

  switch (data.type) {
    case 'thinking':
      renderThinking(data.content);
      break;
    case 'text':
      // Ensure thinking is finalized before text
      if (activeThinkCard) finalizeThinkCard();
      renderText(data.content);
      break;
    case 'tool_call':
      finalizeTextCard();
      finalizeThinkCard();
      renderToolCall(data.tool_name, data.content);
      stats.tools++;
      updateStats();
      break;
    case 'tool_result':
      renderToolResult(data.tool_name, data.content);
      break;
    case 'error':
      finalizeTextCard();
      finalizeThinkCard();
      renderError(data.content, data.tool_name);
      break;
    case 'status':
      handleStatus(data.content, data.metadata);
      break;
    case 'done':
      finalizeTextCard();
      finalizeThinkCard();
      setStatus('done', '✅ Done');
      isAgentRunning = false;
      setTypingVisible(false);
      setStopGenVisible(false);
      setBtnLoading(false);
      break;
  }
}

// ── Renderers ────────────────────────────────────────────────────

function renderThinking(text) {
  hideFeedEmpty();
  if (!activeThinkCard) {
    activeThinkCard = createCard('thinking', '🧠', 'Reasoning', '', true);
  }
  const body = activeThinkCard.querySelector('.card-body');
  body.textContent += text;
  autoScroll();
}

function renderText(text) {
  hideFeedEmpty();
  if (!activeTextCard) {
    activeTextCard = createCard('text', '💬', 'Agent', '', true);
    const body = activeTextCard.querySelector('.card-body');
    body.innerHTML = '<div class="md-content streaming-cursor"></div>';
  }
  const mdDiv = activeTextCard.querySelector('.md-content');
  mdDiv._raw = (mdDiv._raw || '') + text;
  mdDiv.innerHTML = marked.parse(mdDiv._raw);
  mdDiv.classList.add('streaming-cursor');
  // Re-highlight code blocks
  mdDiv.querySelectorAll('pre code').forEach(block => {
    if (!block.dataset.highlighted) {
      hljs.highlightElement(block);
      block.dataset.highlighted = 'yes';
    }
  });
  autoScroll();
}

function finalizeTextCard() {
  if (activeTextCard) {
    const mdDiv = activeTextCard.querySelector('.md-content');
    if (mdDiv) {
      mdDiv.classList.remove('streaming-cursor');
      // Final highlight pass
      mdDiv.querySelectorAll('pre code:not([data-highlighted])').forEach(b => hljs.highlightElement(b));
    }
    activeTextCard = null;
  }
}

function finalizeThinkCard() {
  activeThinkCard = null;
}

function renderToolCall(toolName, args) {
  hideFeedEmpty();
  stats.turns++;
  updateStats();

  const card = createCard('tool-call', '🔧', 'Tool Call', toolName, true);
  const body = card.querySelector('.card-body');

  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.classList.add('language-json');
  // Try to pretty-print JSON args
  try {
    const parsed = JSON.parse(args);
    code.textContent = JSON.stringify(parsed, null, 2);
  } catch {
    code.textContent = args || '(no arguments)';
  }
  pre.appendChild(code);
  body.appendChild(pre);
  hljs.highlightElement(code);

  autoScroll();
}

function renderToolResult(toolName, output) {
  hideFeedEmpty();
  const card = createCard('tool-result', '📤', 'Tool Output', toolName, false);
  const body = card.querySelector('.card-body');
  body.textContent = output || '(empty)';
  autoScroll();
}

function renderError(msg, toolName) {
  hideFeedEmpty();
  const card = createCard('error', '⚠️', 'Error', toolName || '', true);
  const body = card.querySelector('.card-body');
  body.textContent = msg;
  autoScroll();
}

function renderUserMsg(text) {
  finalizeTextCard();
  finalizeThinkCard();
  hideFeedEmpty();
  const card = createCard('user', '🙋', 'You', '', true);
  const body = card.querySelector('.card-body');
  body.textContent = text;
  autoScroll();
}

// ── Card Factory ────────────────────────────────────────────────
function createCard(type, icon, label, toolName, openByDefault) {
  const card = document.createElement('div');
  card.className = `event-card ${type}`;

  const header = document.createElement('div');
  header.className = 'card-header';
  header.innerHTML = `
    <span class="card-icon">${icon}</span>
    <span class="card-label">${label}</span>
    ${toolName ? `<span class="card-tool-name">${toolName}</span>` : ''}
    <span class="card-toggle ${openByDefault ? 'open' : ''}">▼</span>
  `;

  const body = document.createElement('div');
  body.className = `card-body ${openByDefault ? 'open' : ''}`;

  header.addEventListener('click', () => {
    const open = body.classList.toggle('open');
    header.querySelector('.card-toggle').classList.toggle('open', open);
  });

  card.appendChild(header);
  card.appendChild(body);
  document.getElementById('feed').appendChild(card);
  return card;
}

// ── Status Handling ──────────────────────────────────────────────
function handleStatus(content, metadata) {
  if (content?.includes('Starting agent')) {
    setStatus('running', 'Running');
    isAgentRunning = true;
    setTypingVisible(true);
    setStopGenVisible(true);
  } else if (content?.includes('cancelled')) {
    setStatus('idle', 'Idle');
    isAgentRunning = false;
    setTypingVisible(false);
    setStopGenVisible(false);
    setBtnLoading(false);
  }

  if (metadata?.done) {
    setStatus('done', '✅ Done');
    isAgentRunning = false;
    setTypingVisible(false);
    setStopGenVisible(false);
    setBtnLoading(false);
  }

  // Only show non-trivial status messages
  if (content && content !== 'Calling model...') {
    const feed = document.getElementById('feed');
    const el = document.createElement('div');
    el.className = 'event-status';
    el.textContent = content;
    feed.appendChild(el);
    autoScroll();
  }
}

// ── API Calls ────────────────────────────────────────────────────
async function startAgent() {
  if (isAgentRunning) return;
  isAgentRunning = true;
  setStatus('running', 'Starting...');
  setTypingVisible(true);
  setStopGenVisible(true);
  setBtnLoading(true);

  try {
    const res = await fetch('/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'Begin solving the competition described in competition.md. Take full ownership and win.' }),
    });
    const data = await res.json();
    if (data.error) {
      renderError(data.error);
      setStatus('error', 'Error');
      isAgentRunning = false;
      setStopGenVisible(false);
      setBtnLoading(false);
    }
  } catch (e) {
    renderError(`Failed to start agent: ${e.message}`);
    setStatus('error', 'Error');
    isAgentRunning = false;
    setStopGenVisible(false);
    setBtnLoading(false);
  }
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  finalizeTextCard();
  finalizeThinkCard();
  renderUserMsg(text);

  setTypingVisible(true);
  setStopGenVisible(true);
  setStatus('running', 'Responding...');

  try {
    await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
  } catch (e) {
    renderError(`Chat failed: ${e.message}`);
    setStopGenVisible(false);
  }
}

async function stopGeneration() {
  try {
    await fetch('/reset', { method: 'POST' });
    finalizeTextCard();
    finalizeThinkCard();
    setTypingVisible(false);
    setStopGenVisible(false);
    setStatus('idle', 'Stopped');
    isAgentRunning = false;
    setBtnLoading(false);
  } catch (e) {
    console.warn('Stop failed:', e);
  }
}

async function resetAgent() {
  if (!confirm('Full reset? This clears all conversation, stops all processes, and reloads the agent internally.')) return;

  // Reset UI immediately
  const feed = document.getElementById('feed');
  feed.innerHTML = '';
  const emptyEl = document.createElement('div');
  emptyEl.className = 'feed-empty';
  emptyEl.id = 'feed-empty';
  emptyEl.innerHTML = `
    <div class="empty-icon">⚡</div>
    <div class="empty-title">KaggleClaw Ready</div>
    <div class="empty-body">Agent has been reset. Fill in <code>competition.md</code> and click <strong>Start Agent</strong>.</div>
  `;
  feed.appendChild(emptyEl);

  stats = { turns: 0, tools: 0, events: 0 };
  updateStats();
  isAgentRunning = false;
  finalizeTextCard();
  finalizeThinkCard();
  setStatus('idle', 'Idle');
  setTypingVisible(false);
  setStopGenVisible(false);
  setBtnLoading(false);

  try {
    await fetch('/reset', { method: 'POST' });
  } catch (e) {
    console.warn('Reset failed:', e);
  }

  // Reconnect SSE to fresh queue
  connectSSE();
}

// ── File Tree ────────────────────────────────────────────────────
async function loadFileTree() {
  const treeEl = document.getElementById('file-tree');
  const loadingEl = document.getElementById('file-tree-loading');
  try {
    const res = await fetch('/files');
    if (!res.ok) throw new Error('No /files endpoint');
    const data = await res.json();
    if (loadingEl) loadingEl.style.display = 'none';
    renderFileTree(treeEl, data.tree || []);
  } catch (e) {
    if (loadingEl) loadingEl.textContent = 'File tree unavailable';
  }
}

function refreshFileTree() {
  const treeEl = document.getElementById('file-tree');
  const loadingEl = document.getElementById('file-tree-loading');
  treeEl.innerHTML = '';
  if (loadingEl) { loadingEl.style.display = ''; loadingEl.textContent = 'Loading...'; }
  loadFileTree();
}

function renderFileTree(container, nodes, depth = 0) {
  for (const node of nodes) {
    if (node.type === 'dir') {
      const wrapper = document.createElement('div');
      const dirEl = document.createElement('div');
      dirEl.className = 'ft-dir';
      dirEl.innerHTML = `<span class="ft-dir-arrow">▶</span>📁 ${node.name}`;

      const children = document.createElement('div');
      children.className = 'ft-children hidden';
      if (node.children && node.children.length > 0) {
        renderFileTree(children, node.children, depth + 1);
      }

      dirEl.addEventListener('click', () => {
        children.classList.toggle('hidden');
        dirEl.querySelector('.ft-dir-arrow').classList.toggle('open');
      });

      wrapper.appendChild(dirEl);
      wrapper.appendChild(children);
      container.appendChild(wrapper);
    } else {
      const fileEl = document.createElement('div');
      fileEl.className = 'ft-file';
      const ext = node.name.split('.').pop();
      const icon = getFileIcon(ext);
      fileEl.innerHTML = `${icon} ${node.name}`;
      container.appendChild(fileEl);
    }
  }
}

function getFileIcon(ext) {
  const icons = {
    py: '🐍', js: '📜', json: '📋', md: '📝', csv: '📊',
    txt: '📄', html: '🌐', css: '🎨', ipynb: '📓', sh: '⚙️'
  };
  return icons[ext] || '📄';
}

// ── Sidebar Load ─────────────────────────────────────────────────
async function loadCompetitionInfo() {
  try {
    const res = await fetch('/competition');
    const data = await res.json();
    setText('s-name', data.name || '—');
    setText('s-task', data.task || '—');
    setText('s-metric', data.metric || '—');
    setText('s-target', data.target || '—');
    setText('s-deadline', data.deadline || '—');
    setText('comp-task', data.task || '—');
    setText('comp-metric', data.metric || '—');

    if (data.url && data.url !== '—') {
      const link = document.getElementById('s-url');
      link.href = data.url;
    }

    const badge = document.getElementById('comp-badge');
    if (data.task !== '—' || data.metric !== '—') {
      badge.classList.remove('hidden');
    }
  } catch (e) {
    console.warn('Could not load competition info:', e);
  }
}

async function loadHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.public_url) {
      const pill = document.getElementById('url-pill');
      const urlText = document.getElementById('url-text');
      pill.href = data.public_url;
      urlText.textContent = data.public_url.replace('https://', '');
      pill.classList.remove('hidden');
    }
  } catch (e) {
    // fine
  }
}

// ── UI Helpers ───────────────────────────────────────────────────
function hideFeedEmpty() {
  const el = document.getElementById('feed-empty');
  if (el) el.style.display = 'none';
}

function setStatus(type, text) {
  const pill = document.getElementById('status-pill');
  pill.className = `status-pill ${type}`;
  document.getElementById('status-text').textContent = text;
}

function setTypingVisible(visible) {
  const el = document.getElementById('typing-indicator');
  el.classList.toggle('hidden', !visible);
}

function setStopGenVisible(visible) {
  const el = document.getElementById('btn-stop-gen');
  el.classList.toggle('hidden', !visible);
}

function setBtnLoading(loading) {
  const btn = document.getElementById('btn-start');
  btn.disabled = loading;
  btn.textContent = loading ? '⏳ Running...' : '▶ Start Agent';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function updateStats() {
  setText('stat-turns', stats.turns);
  setText('stat-tools', stats.tools);
  setText('stat-events', stats.events);
}

function autoScroll() {
  const feed = document.getElementById('feed');
  // Only auto-scroll if user is near the bottom
  const threshold = 120;
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < threshold;
  if (nearBottom) {
    feed.scrollTop = feed.scrollHeight;
  }
}

// ── Model Hosting ────────────────────────────────────────────────
async function triggerHostModel() {
  const btn = document.getElementById('btn-host-model');
  const btnStop = document.getElementById('btn-stop-hosting');
  btn.disabled = true;
  btn.textContent = '⏳ Hosting...';

  try {
    const res = await fetch('/host_model', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'hosting_started' || data.status === 'already_hosting') {
      btn.classList.add('hidden');
      if (btnStop) btnStop.classList.remove('hidden');
    } else {
      btn.textContent = '❌ Failed';
      btn.disabled = false;
      console.error(data.error || 'Unknown error');
    }
  } catch (e) {
    btn.textContent = '❌ Failed';
    btn.disabled = false;
    console.error(e);
  }
}

async function triggerStopHosting() {
  const btnStart = document.getElementById('btn-host-model');
  const btnStop = document.getElementById('btn-stop-hosting');
  btnStop.disabled = true;
  btnStop.textContent = '⏳ Stopping...';

  try {
    await fetch('/stop_hosting', { method: 'POST' });
    btnStop.classList.add('hidden');
    btnStop.disabled = false;
    btnStop.textContent = 'Stop Hosting';

    if (btnStart) {
      btnStart.classList.remove('hidden');
      btnStart.disabled = false;
      btnStart.textContent = 'Host Model';
    }
  } catch (e) {
    btnStop.textContent = '❌ Failed';
    btnStop.disabled = false;
    console.error(e);
  }
}

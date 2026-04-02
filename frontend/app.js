/**
 * KaggleClaw — Frontend SSE consumer and event renderer
 * ES6 vanilla JS, no frameworks
 */

// ── State ──────────────────────────────────────────────────────
let eventSource = null;
let stats = { turns: 0, tools: 0, events: 0 };
let isAgentRunning = false;

// Streaming text accumulation
let activeTextCard = null;
let activeThinkCard = null;

// ── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadCompetitionInfo();
  loadHealth();
  connectSSE();
  hljs.highlightAll();
});

// ── SSE Connection ─────────────────────────────────────────────
function connectSSE() {
  if (eventSource) { eventSource.close(); }

  eventSource = new EventSource('/stream');

  eventSource.onopen = () => {
    console.log('[KaggleClaw] SSE connected');
  };

  eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    handleEvent(data);
  };

  eventSource.onerror = () => {
    console.warn('[KaggleClaw] SSE connection lost, retrying...');
    setTimeout(connectSSE, 3000);
  };
}

// ── Event Router ───────────────────────────────────────────────
function handleEvent(data) {
  if (data.type === 'ping') return;

  stats.events++;
  updateStats();

  switch (data.type) {
    case 'thinking':
      renderThinking(data.content);
      break;
    case 'text':
      renderText(data.content);
      break;
    case 'tool_call':
      activeTextCard = null;
      activeThinkCard = null;
      renderToolCall(data.tool_name, data.content);
      stats.tools++;
      updateStats();
      break;
    case 'tool_result':
      renderToolResult(data.tool_name, data.content);
      break;
    case 'error':
      activeTextCard = null;
      activeThinkCard = null;
      renderError(data.content, data.tool_name);
      break;
    case 'status':
      handleStatus(data.content, data.metadata);
      break;
    case 'done':
      setStatus('done', 'Done');
      isAgentRunning = false;
      setTypingVisible(false);
      setBtnLoading(false);
      activeTextCard = null;
      activeThinkCard = null;
      break;
  }
}

// ── Renderers ──────────────────────────────────────────────────

function renderThinking(text) {
  hideFeedEmpty();
  if (!activeThinkCard) {
    activeThinkCard = createCard('thinking', '🧠', 'Thinking', '', true);
  }
  // Append to body
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
  // accumulate raw text, re-render markdown
  mdDiv._raw = (mdDiv._raw || '') + text;
  mdDiv.innerHTML = marked.parse(mdDiv._raw);
  mdDiv.classList.add('streaming-cursor');
  hljs.highlightAll();
  autoScroll();
}

function finalizeTextCard() {
  if (activeTextCard) {
    const mdDiv = activeTextCard.querySelector('.md-content');
    if (mdDiv) mdDiv.classList.remove('streaming-cursor');
    activeTextCard = null;
  }
}

function renderToolCall(toolName, args) {
  finalizeTextCard();
  hideFeedEmpty();
  stats.turns++;
  updateStats();

  const card = createCard('tool-call', '🔧', 'Tool Call', toolName, true);
  const body = card.querySelector('.card-body');

  // Syntax-highlighted code block
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.classList.add('language-python');
  code.textContent = args || '(no arguments)';
  pre.appendChild(code);
  body.appendChild(pre);
  hljs.highlightElement(code);

  autoScroll();
}

function renderToolResult(toolName, output) {
  hideFeedEmpty();
  // Default: collapsed
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
  hideFeedEmpty();
  const card = createCard('user', '🙋', 'You', '', true);
  const body = card.querySelector('.card-body');
  body.textContent = text;
  autoScroll();
}

// ── Card Factory ───────────────────────────────────────────────
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

// ── Status handling ────────────────────────────────────────────
function handleStatus(content, metadata) {
  if (content?.includes('Starting agent')) {
    setStatus('running', 'Running');
    isAgentRunning = true;
    setTypingVisible(true);
  } else if (content?.includes('cancelled')) {
    setStatus('idle', 'Idle');
    isAgentRunning = false;
    setTypingVisible(false);
    setBtnLoading(false);
  }

  if (metadata?.done) {
    setStatus('done', '✅ Done');
    isAgentRunning = false;
    setTypingVisible(false);
    setBtnLoading(false);
  }

  // Show status as a subtle inline item (not a full card)
  if (content && content !== 'Calling model...') {
    const feed = document.getElementById('feed');
    const el = document.createElement('div');
    el.className = 'event-status';
    el.textContent = content;
    feed.appendChild(el);
    autoScroll();
  }
}

// ── API Calls ──────────────────────────────────────────────────
async function startAgent() {
  if (isAgentRunning) return;
  isAgentRunning = true;
  setStatus('running', 'Starting...');
  setTypingVisible(true);
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
      setBtnLoading(false);
    }
  } catch (e) {
    renderError(`Failed to start agent: ${e.message}`);
    setStatus('error', 'Error');
    isAgentRunning = false;
    setBtnLoading(false);
  }
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  renderUserMsg(text);

  setTypingVisible(true);
  setStatus('running', 'Responding...');

  try {
    await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
  } catch (e) {
    renderError(`Chat failed: ${e.message}`);
  }
}

async function resetAgent() {
  if (!confirm('Reset the conversation and agent state?')) return;

  // Clear UI
  const feed = document.getElementById('feed');
  feed.innerHTML = '';
  document.getElementById('feed-empty').style.display = '';
  feed.appendChild(document.getElementById('feed-empty'));

  stats = { turns: 0, tools: 0, events: 0 };
  updateStats();
  isAgentRunning = false;
  activeTextCard = null;
  activeThinkCard = null;
  setStatus('idle', 'Idle');
  setTypingVisible(false);
  setBtnLoading(false);

  try {
    await fetch('/reset', { method: 'POST' });
  } catch (e) {
    console.warn('Reset failed:', e);
  }

  // Reconnect SSE
  connectSSE();
}

// ── Sidebar Init ───────────────────────────────────────────────
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
    // not a problem if health fails at load
  }
}

// ── UI Helpers ─────────────────────────────────────────────────
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
  feed.scrollTop = feed.scrollHeight;
}

/**
 * KaggleClaw — Frontend chat consumer (v3)
 * Flat ChatGPT-style interface. No overflow containers causing scroll issues.
 * Each message is a flex row: avatar + content bubble.
 */

// ── State ────────────────────────────────────────────────────────
let eventSource = null;
let stats = { turns: 0, tools: 0, events: 0 };
let isAgentRunning = false;

// Active streaming text node for the current assistant turn
let activeAssistantMsg = null; // DOM element for the current streaming message
let activeThinkBlock = null;   // DOM element for the current thinking block

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  marked.setOptions({ breaks: true, gfm: true });
  loadCompetitionInfo();
  loadHealth();
  connectSSE();
  loadFileTree();
});

// ── SSE ──────────────────────────────────────────────────────────
function connectSSE() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/stream');
  eventSource.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); } catch {}
  };
  eventSource.onerror = () => setTimeout(connectSSE, 3000);
}

// ── Event Router ─────────────────────────────────────────────────
function handleEvent(data) {
  if (data.type === 'ping') return;
  stats.events++;
  updateStats();

  switch (data.type) {
    case 'thinking':
      appendThinking(data.content);
      break;
    case 'text':
      finalizeThinking();
      appendAssistantText(data.content);
      break;
    case 'tool_call':
      finalizeAssistant();
      finalizeThinking();
      appendToolCall(data.tool_name, data.content);
      stats.tools++;
      updateStats();
      break;
    case 'tool_result':
      appendToolResult(data.tool_name, data.content);
      break;
    case 'error':
      finalizeAssistant();
      finalizeThinking();
      appendError(data.content);
      break;
    case 'status':
      handleStatusEvent(data.content, data.metadata);
      break;
    case 'done':
      finalizeAssistant();
      finalizeThinking();
      setStatus('done', '✅ Done');
      isAgentRunning = false;
      setTypingVisible(false);
      setStopGenVisible(false);
      setBtnLoading(false);
      break;
  }
}

// ── Chat Message Builders ────────────────────────────────────────

/** Create a message row. Returns the content element. */
function createMsgRow(type, icon, headerLabel) {
  const row = document.createElement('div');
  row.className = `msg-row msg-${type}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = icon;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (headerLabel) {
    const hdr = document.createElement('div');
    hdr.className = 'msg-header';
    hdr.textContent = headerLabel;
    bubble.appendChild(hdr);
  }

  const body = document.createElement('div');
  body.className = 'msg-body';
  bubble.appendChild(body);

  if (type === 'user') {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  document.getElementById('feed').appendChild(row);
  autoScroll();
  hideFeedEmpty();
  return body;
}

// ── Thinking block (collapsible) ─────────────────────────────────
function appendThinking(text) {
  if (!activeThinkBlock) {
    // Create the thinking collapsible row
    const row = document.createElement('div');
    row.className = 'msg-row msg-think';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🧠';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const toggle = document.createElement('div');
    toggle.className = 'think-toggle';
    toggle.innerHTML = '<span class="think-label">Reasoning <span class="think-arrow">▾</span></span>';
    const body = document.createElement('div');
    body.className = 'think-body open';
    body.style.whiteSpace = 'pre-wrap';

    toggle.addEventListener('click', () => {
      body.classList.toggle('open');
      toggle.querySelector('.think-arrow').textContent = body.classList.contains('open') ? '▾' : '▸';
    });

    bubble.appendChild(toggle);
    bubble.appendChild(body);
    row.appendChild(avatar);
    row.appendChild(bubble);
    document.getElementById('feed').appendChild(row);
    activeThinkBlock = body;
    hideFeedEmpty();
  }
  activeThinkBlock.textContent += text;
  autoScroll();
}

function finalizeThinking() {
  activeThinkBlock = null;
}

// ── Assistant text (streaming markdown) ──────────────────────────
function appendAssistantText(text) {
  if (!activeAssistantMsg) {
    const row = document.createElement('div');
    row.className = 'msg-row msg-assistant';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '⚡';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const body = document.createElement('div');
    body.className = 'msg-body md-content streaming-cursor';
    body._raw = '';
    bubble.appendChild(body);

    row.appendChild(avatar);
    row.appendChild(bubble);
    document.getElementById('feed').appendChild(row);
    activeAssistantMsg = body;
    hideFeedEmpty();
  }

  activeAssistantMsg._raw = (activeAssistantMsg._raw || '') + text;
  activeAssistantMsg.innerHTML = marked.parse(activeAssistantMsg._raw);
  activeAssistantMsg.classList.add('streaming-cursor');

  // Re-highlight code blocks without duplicating
  activeAssistantMsg.querySelectorAll('pre code:not([data-hl])').forEach(block => {
    hljs.highlightElement(block);
    block.dataset.hl = '1';
  });
  autoScroll();
}

function finalizeAssistant() {
  if (activeAssistantMsg) {
    activeAssistantMsg.classList.remove('streaming-cursor');
    activeAssistantMsg.querySelectorAll('pre code:not([data-hl])').forEach(block => {
      hljs.highlightElement(block);
      block.dataset.hl = '1';
    });
    activeAssistantMsg = null;
  }
}

// ── User message ──────────────────────────────────────────────────
function appendUserMsg(text) {
  finalizeAssistant();
  finalizeThinking();
  const body = createMsgRow('user', '🙋', null);
  body.textContent = text;
}

// ── Tool Call ─────────────────────────────────────────────────────
function appendToolCall(toolName, args) {
  stats.turns++;
  updateStats();

  const row = document.createElement('div');
  row.className = 'msg-row msg-tool';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '🔧';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble tool-bubble';

  const hdr = document.createElement('div');
  hdr.className = 'msg-header tool-header';
  hdr.innerHTML = `Tool Call <span class="tool-name">${toolName}</span>`;

  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.classList.add('language-json');
  try {
    code.textContent = JSON.stringify(JSON.parse(args), null, 2);
  } catch { 
    code.textContent = args || '{}'; 
  }
  pre.appendChild(code);
  hljs.highlightElement(code);

  bubble.appendChild(hdr);
  bubble.appendChild(pre);
  row.appendChild(avatar);
  row.appendChild(bubble);
  document.getElementById('feed').appendChild(row);
  hideFeedEmpty();
  autoScroll();
}

// ── Tool Result ───────────────────────────────────────────────────
function appendToolResult(toolName, output) {
  const row = document.createElement('div');
  row.className = 'msg-row msg-tool-result';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '📤';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  const toggle = document.createElement('div');
  toggle.className = 'think-toggle';
  toggle.innerHTML = `<span class="think-label">Output <span class="tool-name">${toolName}</span> <span class="think-arrow">▸</span></span>`;

  const body = document.createElement('pre');
  body.className = 'tool-output';
  body.style.display = 'none';
  body.textContent = output || '(empty)';

  toggle.addEventListener('click', () => {
    const open = body.style.display === 'none';
    body.style.display = open ? '' : 'none';
    toggle.querySelector('.think-arrow').textContent = open ? '▾' : '▸';
  });

  bubble.appendChild(toggle);
  bubble.appendChild(body);
  row.appendChild(avatar);
  row.appendChild(bubble);
  document.getElementById('feed').appendChild(row);
  autoScroll();
}

// ── Error ─────────────────────────────────────────────────────────
function appendError(msg) {
  const body = createMsgRow('error', '⚠️', 'Error');
  body.textContent = msg;
}

// ── Status Event ──────────────────────────────────────────────────
function handleStatusEvent(content, metadata) {
  if (!content || content === 'Calling model...') return;

  if (content.includes('Starting agent') || content.includes('Turn ')) {
    setStatus('running', 'Running');
    isAgentRunning = true;
    setTypingVisible(true);
    setStopGenVisible(true);
  } else if (content.includes('cancelled')) {
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

  // Small inline status message
  const feed = document.getElementById('feed');
  const el = document.createElement('div');
  el.className = 'status-chip';
  el.textContent = content;
  feed.appendChild(el);
  autoScroll();
}

// ── API Calls ─────────────────────────────────────────────────────
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
      appendError(data.error);
      setStatus('error', 'Error');
      isAgentRunning = false;
      setStopGenVisible(false);
      setBtnLoading(false);
    }
  } catch (e) {
    appendError(`Failed to start agent: ${e.message}`);
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

  appendUserMsg(text);
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
    appendError(`Chat failed: ${e.message}`);
    setStopGenVisible(false);
  }
}

async function stopGeneration() {
  try {
    await fetch('/reset', { method: 'POST' });
    finalizeAssistant();
    finalizeThinking();
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
  if (!confirm('Full reset? Clears conversation, stops all processes.')) return;

  const feed = document.getElementById('feed');
  feed.innerHTML = `
    <div class="feed-empty" id="feed-empty">
      <div class="empty-icon">⚡</div>
      <div class="empty-title">KaggleClaw Ready</div>
      <div class="empty-body">Agent reset. Fill in <code>competition.md</code> and click <strong>Start Agent</strong>.</div>
    </div>
  `;

  stats = { turns: 0, tools: 0, events: 0 };
  updateStats();
  isAgentRunning = false;
  activeAssistantMsg = null;
  activeThinkBlock = null;
  setStatus('idle', 'Idle');
  setTypingVisible(false);
  setStopGenVisible(false);
  setBtnLoading(false);

  try { await fetch('/reset', { method: 'POST' }); } catch {}
  connectSSE();
}

// ── File Tree ─────────────────────────────────────────────────────
async function loadFileTree() {
  const treeEl = document.getElementById('file-tree');
  const loadingEl = document.getElementById('file-tree-loading');
  try {
    const res = await fetch('/files');
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (loadingEl) loadingEl.style.display = 'none';
    renderFileTree(treeEl, data.tree || []);
  } catch {
    if (loadingEl) loadingEl.textContent = 'Unavailable';
  }
}

function refreshFileTree() {
  const treeEl = document.getElementById('file-tree');
  const loadingEl = document.getElementById('file-tree-loading');
  treeEl.innerHTML = '';
  if (loadingEl) { loadingEl.style.display = ''; loadingEl.textContent = 'Loading...'; }
  loadFileTree();
}

function renderFileTree(container, nodes) {
  for (const node of nodes) {
    if (node.type === 'dir') {
      const wrap = document.createElement('div');
      const dirEl = document.createElement('div');
      dirEl.className = 'ft-dir';
      dirEl.innerHTML = `<span class="ft-arrow">▸</span>📁 ${node.name}`;

      const children = document.createElement('div');
      children.className = 'ft-children hidden';
      if (node.children?.length) renderFileTree(children, node.children);

      dirEl.addEventListener('click', () => {
        children.classList.toggle('hidden');
        dirEl.querySelector('.ft-arrow').textContent = children.classList.contains('hidden') ? '▸' : '▾';
      });
      wrap.appendChild(dirEl);
      wrap.appendChild(children);
      container.appendChild(wrap);
    } else {
      const fileEl = document.createElement('div');
      fileEl.className = 'ft-file';
      const ext = node.name.split('.').pop();
      const icons = { py:'🐍',js:'📜',json:'📋',md:'📝',csv:'📊',txt:'📄',html:'🌐',css:'🎨',ipynb:'📓',sh:'⚙️' };
      fileEl.innerHTML = `${icons[ext] || '📄'} ${node.name}`;
      container.appendChild(fileEl);
    }
  }
}

// ── Sidebar Data ──────────────────────────────────────────────────
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
    if (data.url && data.url !== '—') document.getElementById('s-url').href = data.url;
    if (data.task !== '—' || data.metric !== '—') document.getElementById('comp-badge').classList.remove('hidden');
  } catch {}
}

async function loadHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.public_url) {
      const pill = document.getElementById('url-pill');
      document.getElementById('url-text').textContent = data.public_url.replace('https://', '');
      pill.href = data.public_url;
      pill.classList.remove('hidden');
    }
  } catch {}
}

// ── UI Helpers ────────────────────────────────────────────────────
function hideFeedEmpty() {
  const el = document.getElementById('feed-empty');
  if (el) el.style.display = 'none';
}

function setStatus(type, text) {
  const pill = document.getElementById('status-pill');
  pill.className = `status-pill ${type}`;
  document.getElementById('status-text').textContent = text;
}

function setTypingVisible(v) {
  document.getElementById('typing-indicator').classList.toggle('hidden', !v);
}

function setStopGenVisible(v) {
  document.getElementById('btn-stop-gen').classList.toggle('hidden', !v);
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
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 150;
  if (nearBottom) feed.scrollTop = feed.scrollHeight;
}

// ── Model Hosting ─────────────────────────────────────────────────
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
      btn.textContent = '❌ Failed'; btn.disabled = false;
    }
  } catch { btn.textContent = '❌ Failed'; btn.disabled = false; }
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
    if (btnStart) { btnStart.classList.remove('hidden'); btnStart.disabled = false; btnStart.textContent = 'Host Model'; }
  } catch { btnStop.textContent = '❌ Failed'; btnStop.disabled = false; }
}

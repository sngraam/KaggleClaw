/**
 * KaggleClaw — Frontend (v4)
 * Features:
 *   • Harmony special token decoder (200002–200012)
 *   • Throttled streaming render via requestAnimationFrame
 *   • KaTeX LaTeX rendering ($...$ and $$...$$) after markdown
 *   • Collapsible debug event log panel
 *   • Detailed error display with stack traces
 */

// ── Harmony Special Tokens ───────────────────────────────────────────────────
const HARMONY_TOKENS = {
  200002: { label: '<|return|>',    color: '#f87171' },
  200003: { label: '<|constrain|>', color: '#a78bfa' },
  200005: { label: '<|channel|>',   color: '#60a5fa' },
  200006: { label: '<|start|>',     color: '#34d399' },
  200007: { label: '<|end|>',       color: '#fb923c' },
  200008: { label: '<|message|>',   color: '#facc15' },
  200012: { label: '<|call|>',      color: '#f472b6' },
};

// ── State ─────────────────────────────────────────────────────────────────────
let eventSource      = null;
let stats            = { turns: 0, tools: 0, events: 0 };
let isAgentRunning   = false;
let debugVisible     = false;

// Active streaming elements
let activeAssistantMsg = null;
let activeThinkBlock   = null;
let activeToolCall     = null;

// Throttled render queue
let _renderQueue    = [];   // [{el, rawText}]
let _rafScheduled   = false;
let _debugLogEl     = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  marked.setOptions({ breaks: true, gfm: true });
  _debugLogEl = document.getElementById('debug-log');
  loadCompetitionInfo();
  loadHealth();
  connectSSE();
  loadFileTree();
  loadServerLogs();  // populate terminal logger from server ring-buffer on load
});

// ── KaTeX helper ──────────────────────────────────────────────────────────────
function renderMathInAllElements() { /* called by KaTeX auto-render onload */ }

function renderMathIn(el) {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$',  right: '$',  display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
      });
    } catch (_) {}
  }
}

// ── SSE Connection ────────────────────────────────────────────────────────────
function connectSSE() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/stream');
  eventSource.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); } catch(err) { logDebug('parse_error', err.message, '#f87171'); }
  };
  eventSource.onerror = () => setTimeout(connectSSE, 3000);
}

// ── Event Router ──────────────────────────────────────────────────────────────
function handleEvent(data) {
  if (data.type === 'ping') return;   // SSE keep-alive — ignore silently
  stats.events++;
  updateStats();

  // Debug log every event
  const tokenLabel = data.metadata?.token || '';
  logDebug(data.type, (data.content || '').slice(0, 120) + (data.tool_name ? ` [${data.tool_name}]` : '') + (tokenLabel ? ` ${tokenLabel}` : ''));

  // Terminal logger — pipe key events
  switch (data.type) {
    case 'status':     appendTerminalLog('status',  data.content); break;
    case 'error':      appendTerminalLog('error',   data.content); break;
    case 'tool_call':  appendTerminalLog('tool',    `▶ ${data.tool_name}: ${(data.content||'').slice(0,120)}`); break;
    case 'tool_result':appendTerminalLog('result',  `◀ ${data.tool_name}: ${(data.content||'').slice(0,120)}`); break;
    case 'done':       appendTerminalLog('done',    data.content || '✅ Agent done'); break;
    case 'thinking':   appendTerminalLog('think',   `🧠 ${(data.content||'').slice(0,80)}`); break;
    default: break;
  }

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
      break;
    case 'tool_result':
      finalizeToolCall();
      appendToolResult(data.tool_name, data.content);
      break;
    case 'error':
      finalizeAssistant();
      finalizeThinking();
      finalizeToolCall();
      appendError(data.content);
      break;
    case 'status':
      handleStatusEvent(data.content, data.metadata);
      break;
    case 'done':
      finalizeAssistant();
      finalizeThinking();
      finalizeToolCall();
      setStatus('done', '✅ Done');
      isAgentRunning = false;
      setTypingVisible(false);
      setStopGenVisible(false);
      setBtnLoading(false);
      break;
  }
}

// ── Thinking block (collapsible, streaming) ───────────────────────────────────
function appendThinking(text) {
  if (!activeThinkBlock) {
    const row    = document.createElement('div');
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
  // Throttled append to thinking block
  _enqueueRender(activeThinkBlock, text, 'think');
}

function finalizeThinking() {
  if (activeThinkBlock) {
    _flushRender(activeThinkBlock);
  }
  activeThinkBlock = null;
}

function finalizeToolCall() {
  if (activeToolCall) {
    _flushRender(activeToolCall.el);
    delete activeToolCall.el.dataset.highlighted;
    try { hljs.highlightElement(activeToolCall.el); } catch (e) {}
  }
  activeToolCall = null;
}

// ── Assistant text (streaming markdown + KaTeX) ───────────────────────────────
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
  _enqueueRender(activeAssistantMsg, text, 'markdown');
}

function finalizeAssistant() {
  if (activeAssistantMsg) {
    _flushRender(activeAssistantMsg);
    activeAssistantMsg.classList.remove('streaming-cursor');
    _applyMarkdown(activeAssistantMsg);
    activeAssistantMsg = null;
  }
}

// ── Throttled render system ───────────────────────────────────────────────────

/**
 * Queue a text chunk for a given element.
 * mode: 'think' (plain pre-wrap) | 'markdown' (markdown+katex)
 */
function _enqueueRender(el, text, mode) {
  // Find existing entry for this element
  const existing = _renderQueue.find(q => q.el === el);
  if (existing) {
    existing.text += text;
  } else {
    _renderQueue.push({ el, text, mode });
  }
  if (!_rafScheduled) {
    _rafScheduled = true;
    requestAnimationFrame(_processRenderQueue);
  }
}

function _processRenderQueue() {
  _rafScheduled = false;
  const queue = _renderQueue.splice(0);
  for (const { el, text, mode } of queue) {
    if (mode === 'think') {
      el.textContent += text;
    } else {
      el._raw = (el._raw || '') + text;
      // Only do a quick innerHTML update while streaming; full render on finalize
      el.innerHTML = marked.parse(el._raw);
      el.querySelectorAll('pre code:not([data-hl])').forEach(block => {
        hljs.highlightElement(block);
        block.dataset.hl = '1';
      });
    }
  }
  autoScroll();
  // If still streaming, reschedule
  if (_renderQueue.length > 0 && !_rafScheduled) {
    _rafScheduled = true;
    requestAnimationFrame(_processRenderQueue);
  }
}

function _flushRender(el) {
  const idx = _renderQueue.findIndex(q => q.el === el);
  if (idx >= 0) {
    const { text, mode } = _renderQueue.splice(idx, 1)[0];
    if (mode === 'think') {
      el.textContent += text;
    } else {
      el._raw = (el._raw || '') + text;
    }
  }
}

function _applyMarkdown(el) {
  if (!el._raw) return;
  el.innerHTML = marked.parse(el._raw);
  el.querySelectorAll('pre code:not([data-hl])').forEach(block => {
    hljs.highlightElement(block);
    block.dataset.hl = '1';
  });
  renderMathIn(el);
}

// ── User message ──────────────────────────────────────────────────────────────
function appendUserMsg(text) {
  finalizeAssistant();
  finalizeThinking();
  finalizeToolCall();
  const body = createMsgRow('user', '🙋', null);
  body.textContent = text;
}

// ── Tool Call ─────────────────────────────────────────────────────────────────
function appendToolCall(toolName, args) {
  if (!activeToolCall || activeToolCall.toolName !== toolName) {
    stats.turns++;
    stats.tools++;
    updateStats();

    const row = document.createElement('div');
    row.className = 'msg-row msg-tool';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = _toolIcon(toolName);

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble tool-bubble';

    const hdr = document.createElement('div');
    hdr.className = 'msg-header tool-header';
    hdr.innerHTML = `Tool Call <span class="tool-name">${toolName}</span>`;

    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.classList.add('language-json');
    code.textContent = '';
    pre.appendChild(code);

    bubble.appendChild(hdr);
    bubble.appendChild(pre);
    row.appendChild(avatar);
    row.appendChild(bubble);
    document.getElementById('feed').appendChild(row);
    hideFeedEmpty();

    activeToolCall = { el: code, toolName, content: '' };
  }

  activeToolCall.content += (args || '');
  _enqueueRender(activeToolCall.el, args || '', 'think'); // think mode does pure text append safely
}

// ── Tool Result ───────────────────────────────────────────────────────────────
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

// ── Error message ─────────────────────────────────────────────────────────────
function appendError(msg) {
  const row = document.createElement('div');
  row.className = 'msg-row msg-error';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '⚠️';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.style.background = 'rgba(248,113,113,0.08)';
  bubble.style.border = '1px solid rgba(248,113,113,0.3)';

  const hdr = document.createElement('div');
  hdr.className = 'msg-header';
  hdr.style.color = '#f87171';
  hdr.textContent = '⚠️ Error';

  // Split into short message and optional stack trace
  const lines = (msg || '').split('\n');
  const shortMsg = lines[0] || 'Unknown error';
  const hasTrace = lines.length > 2;

  const body = document.createElement('div');
  body.className = 'msg-body';
  body.style.color = '#fca5a5';
  body.textContent = shortMsg;

  bubble.appendChild(hdr);
  bubble.appendChild(body);

  if (hasTrace) {
    const details = document.createElement('details');
    details.style.marginTop = '8px';
    const summary = document.createElement('summary');
    summary.style.cssText = 'cursor:pointer; font-size:11px; color:#888; list-style:none;';
    summary.textContent = 'Show stack trace ▸';
    const trace = document.createElement('pre');
    trace.style.cssText = 'font-size:11px; color:#aaa; white-space:pre-wrap; margin-top:6px; padding:8px; background:rgba(0,0,0,0.3); border-radius:4px;';
    trace.textContent = lines.slice(1).join('\n');
    details.appendChild(summary);
    details.appendChild(trace);
    bubble.appendChild(details);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  document.getElementById('feed').appendChild(row);
  hideFeedEmpty();
  autoScroll();
}

// ── Status Event ──────────────────────────────────────────────────────────────
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

  const feed = document.getElementById('feed');
  const el = document.createElement('div');
  el.className = 'status-chip';
  el.textContent = content;
  feed.appendChild(el);
  autoScroll();
}

// ── Generic row builder ───────────────────────────────────────────────────────
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

// ── Debug Panel ───────────────────────────────────────────────────────────────
function toggleDebug() {
  debugVisible = !debugVisible;
  const panel = document.getElementById('debug-panel');
  panel.style.display = debugVisible ? 'flex' : 'none';
  document.getElementById('btn-debug').textContent = debugVisible ? '🐛 Hide Debug' : '🐛 Debug';
}

function clearDebug() {
  if (_debugLogEl) _debugLogEl.innerHTML = '';
}

function logDebug(type, content, color) {
  if (!_debugLogEl) return;
  const ts = new Date().toISOString().slice(11, 23);
  const typeColors = {
    thinking:    '#a78bfa',
    text:        '#34d399',
    tool_call:   '#f472b6',
    tool_result: '#60a5fa',
    error:       '#f87171',
    status:      '#facc15',
    done:        '#34d399',
    parse_error: '#f87171',
  };
  const c = color || typeColors[type] || '#a0a0c0';
  const line = document.createElement('div');
  line.style.borderBottom = '1px solid rgba(255,255,255,0.04)';
  line.style.padding = '2px 0';
  line.innerHTML =
    `<span style="color:#555">${ts}</span> ` +
    `<span style="color:${c};font-weight:600">[${type}]</span> ` +
    `<span style="color:#c0c0d0">${_escapeHtml(content)}</span>`;
  _debugLogEl.appendChild(line);
  // Keep max 500 lines
  while (_debugLogEl.children.length > 500) {
    _debugLogEl.removeChild(_debugLogEl.firstChild);
  }
  _debugLogEl.scrollTop = _debugLogEl.scrollHeight;
}

// ── Terminal Logger ───────────────────────────────────────────────────────────

const TERM_COLORS = {
  status:  '#60a5fa',  // blue
  error:   '#f87171',  // red
  tool:    '#f472b6',  // pink
  result:  '#34d399',  // green
  done:    '#a78bfa',  // purple
  think:   '#94a3b8',  // slate
  server:  '#facc15',  // yellow (for server-side logs)
  default: '#c0c0d0',
};

function appendTerminalLog(type, message, isServer = false) {
  const el = document.getElementById('terminal-log');
  if (!el) return;

  const ts  = new Date().toISOString().slice(11, 23);
  const col = TERM_COLORS[type] || TERM_COLORS.default;

  const line = document.createElement('div');
  line.className = 'term-line';
  const prefix = isServer ? '[server] ' : '';
  line.innerHTML =
    `<span class="term-ts">${ts}</span>` +
    `<span class="term-badge" style="color:${col}">[${type}]</span>` +
    `<span class="term-msg">${_escapeHtml(prefix + (message || ''))}</span>`;

  el.appendChild(line);

  // Cap at 800 lines
  while (el.children.length > 800) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

function clearTerminalLog() {
  const el = document.getElementById('terminal-log');
  if (el) el.innerHTML = '';
}

async function copyTerminalLog() {
  const el = document.getElementById('terminal-log');
  if (!el) return;
  const text = Array.from(el.querySelectorAll('.term-line'))
    .map(l => l.textContent)
    .join('\n');
  try {
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById('btn-copy-log');
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = '✅ Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1500);
    }
  } catch (e) {
    // Fallback: open in new window so user can copy manually
    const w = window.open('', '_blank');
    if (w) {
      w.document.write('<pre>' + _escapeHtml(text) + '</pre>');
      w.document.close();
    }
  }
}

async function loadServerLogs() {
  try {
    const res = await fetch('/logs?n=100');
    if (!res.ok) return;
    const data = await res.json();
    for (const line of (data.lines || [])) {
      appendTerminalLog('server', line, true);
    }
  } catch { /* server may not have logs yet */ }
}

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function _toolIcon(name) {
  const icons = {
    python:      '🐍',
    file:        '📁',
    apply_patch: '🩹',
    web_search:  '🔍',
    plan_follow: '📋',
  };
  return icons[name] || '🔧';
}

// ── API Calls ─────────────────────────────────────────────────────────────────
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

  _renderQueue.length = 0;
  _rafScheduled = false;
  stats = { turns: 0, tools: 0, events: 0 };
  updateStats();
  isAgentRunning   = false;
  activeAssistantMsg = null;
  activeThinkBlock   = null;
  activeToolCall     = null;
  setStatus('idle', 'Idle');
  setTypingVisible(false);
  setStopGenVisible(false);
  setBtnLoading(false);
  clearDebug();

  try { await fetch('/reset', { method: 'POST' }); } catch {}
  connectSSE();
}

// ── File Tree ─────────────────────────────────────────────────────────────────
async function loadFileTree() {
  const treeEl    = document.getElementById('file-tree');
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
  const treeEl    = document.getElementById('file-tree');
  const loadingEl = document.getElementById('file-tree-loading');
  treeEl.innerHTML = '';
  if (loadingEl) { loadingEl.style.display = ''; loadingEl.textContent = 'Loading...'; }
  loadFileTree();
}

function renderFileTree(container, nodes) {
  for (const node of nodes) {
    if (node.type === 'dir') {
      const wrap   = document.createElement('div');
      const dirEl  = document.createElement('div');
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

// ── Sidebar Data ──────────────────────────────────────────────────────────────
async function loadCompetitionInfo() {
  try {
    const res  = await fetch('/competition');
    const data = await res.json();
    setText('s-name',    data.name     || '—');
    setText('s-task',    data.task     || '—');
    setText('s-metric',  data.metric   || '—');
    setText('s-target',  data.target   || '—');
    setText('s-deadline',data.deadline || '—');
    setText('comp-task',   data.task   || '—');
    setText('comp-metric', data.metric || '—');
    if (data.url && data.url !== '—') document.getElementById('s-url').href = data.url;
    if (data.task !== '—' || data.metric !== '—') document.getElementById('comp-badge').classList.remove('hidden');
  } catch {}
}

async function loadHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.public_url) {
      const pill = document.getElementById('url-pill');
      document.getElementById('url-text').textContent = data.public_url.replace('https://', '');
      pill.href = data.public_url;
      pill.classList.remove('hidden');
    }
  } catch {}
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
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
  btn.disabled    = loading;
  btn.textContent = loading ? '⏳ Running...' : '▶ Start Agent';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function updateStats() {
  setText('stat-turns',  stats.turns);
  setText('stat-tools',  stats.tools);
  setText('stat-events', stats.events);
}

function autoScroll() {
  const feed = document.getElementById('feed');
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 150;
  if (nearBottom) feed.scrollTop = feed.scrollHeight;
}

// ── Model Hosting ─────────────────────────────────────────────────────────────
async function triggerHostModel() {
  const btn     = document.getElementById('btn-host-model');
  const btnStop = document.getElementById('btn-stop-hosting');
  btn.disabled    = true;
  btn.textContent = '⏳ Hosting...';
  try {
    const res  = await fetch('/host_model', { method: 'POST' });
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
  const btnStop  = document.getElementById('btn-stop-hosting');
  btnStop.disabled    = true;
  btnStop.textContent = '⏳ Stopping...';
  try {
    await fetch('/stop_hosting', { method: 'POST' });
    btnStop.classList.add('hidden');
    btnStop.disabled    = false;
    btnStop.textContent = 'Stop Hosting';
    if (btnStart) { btnStart.classList.remove('hidden'); btnStart.disabled = false; btnStart.textContent = 'Host Model'; }
  } catch { btnStop.textContent = '❌ Failed'; btnStop.disabled = false; }
}

"""
server/app.py — KaggleClaw FastAPI server.

Endpoints:
  GET  /            — serves frontend/index.html
  GET  /stream      — SSE stream of agent events
  POST /start       — start the agent on the competition
  POST /chat        — send a message to the agent
  POST /reset       — reset conversation
  GET  /health      — status + public URL
  POST /host_model  — triggers local vLLM hosting
  POST /stop_hosting — gracefully terminates local vLLM server
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .state import state
from agent.harmo import build_competition_summary

app = FastAPI(title="KaggleClaw", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend files
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize the agent runner on startup."""
    _init_runner()


def _init_runner():
    """Create agent runner with all tools."""
    from agent.run import AgentRunner
    from agent.tools import get_all_tools

    tools = get_all_tools()  # PythonTool, FileTool, ApplyPatchTool, WebSearchTool, PlanFollowTool

    model    = os.environ.get("VLLM_MODEL", "open-scorer-120b")
    base_url = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8080/v1")

    state.runner = AgentRunner(
        event_queue=state.event_queue,
        tools=tools,
        model=model,
        base_url=base_url,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend SPA."""
    index_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>KaggleClaw</h1><p>Frontend not found. Check frontend/ directory.</p>")


@app.get("/health")
async def health():
    return {
        "status": "running",
        "public_url": state.public_url,
        "agent_running": state.runner._running if state.runner else False,
        "turn_count": len(state.runner.messages) if state.runner else 0,
    }


@app.get("/competition")
async def competition_info():
    """Return structured competition info for the frontend sidebar."""
    return build_competition_summary()


@app.get("/files")
async def list_files():
    """Return a JSON directory tree of the Kaggle working directory for the file sidebar."""
    import os as _os

    def _build_tree(path: Path, max_depth: int = 3, depth: int = 0):
        if depth > max_depth:
            return []
        nodes = []
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return []
        for entry in entries:
            if entry.name.startswith('.') or entry.name == '__pycache__':
                continue
            if entry.is_dir():
                nodes.append({
                    "type": "dir",
                    "name": entry.name,
                    "children": _build_tree(entry, max_depth, depth + 1)
                })
            else:
                nodes.append({"type": "file", "name": entry.name})
        return nodes

    working_dir = Path(_os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    if not working_dir.exists():
        # Fallback to project root for local dev
        working_dir = Path(__file__).parent.parent

    return {"base": str(working_dir), "tree": _build_tree(working_dir)}


import json

@app.get("/stream") # (Or whatever your endpoint is named)
async def stream_agent():
    async def event_generator():
        try:
            while True:
                event = await runner.event_queue.get()
                
                # Check if it's our AgentEvent dataclass
                if hasattr(event, "to_sse"):
                    yield event.to_sse()
                else:
                    # Fallback if something else accidentally put a dict in the queue
                    yield f"data: {json.dumps(event)}\n\n"
                
                # Break the loop safely when the agent is done
                if getattr(event, "type", None) == "done" or (isinstance(event, dict) and event.get("type") == "done"):
                    break
                    
        except asyncio.CancelledError:
            # Client disconnected early, just stop safely.
            pass
            
        except Exception as e:
            # ✅ FIX: Yield a properly formatted SSE string, NOT a dict!
            error_payload = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/start")
async def start_agent(request: Request):
    """Start the agent on the current competition."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    message = body.get("message", "Begin solving the competition described in competition.md. Take full ownership.")

    if not state.runner:
        _init_runner()

    if state.agent_task and not state.agent_task.done():
        return JSONResponse({"error": "Agent already running"}, status_code=409)

    state.agent_task = asyncio.create_task(
        state.runner.run(initial_message=message)
    )
    return {"status": "started", "message": message}


@app.post("/chat")
async def chat(request: Request):
    """Send a user message to the agent."""
    body = await request.json()
    message = body.get("message", "").strip()

    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    if not state.runner:
        _init_runner()

    # If agent is idle, start a new turn
    state.agent_task = asyncio.create_task(
        state.runner.send_user_message(message)
    )
    return {"status": "sent", "message": message}


@app.post("/reset")
async def reset():
    """Reset the conversation and agent state."""
    state.reset()
    _init_runner()
    return {"status": "reset"}


@app.post("/host_model")
async def host_model():
    """Start local vLLM server via models/host.py in a background thread to avoid blocking."""
    if state.vllm_server:
        return {"status": "already_hosting"}
    
    import asyncio
    from fastapi.concurrency import run_in_threadpool
    from models.host import Server, ServerConfig

    def _start_server():
        try:
            cfg = ServerConfig()
            state.vllm_server = Server(cfg=cfg, port=8080)
        except Exception as e:
            print(f"Error starting server: {e}")

    # Fire and forget or background task it
    asyncio.create_task(run_in_threadpool(_start_server))
    
    return {"status": "hosting_started"}


@app.post("/stop_hosting")
async def stop_hosting():
    """Stop the local vLLM server if it is running."""
    import subprocess
    if state.vllm_server:
        try:
            # Assumes models/host.py has server_process to kill
            if hasattr(state.vllm_server, "server_process") and state.vllm_server.server_process:
                proc = state.vllm_server.server_process
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as e:
            print(f"Error stopping vLLM: {e}")
        finally:
            state.vllm_server = None
            return {"status": "hosted_stopped"}
    return {"status": "not_hosting"}

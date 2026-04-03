"""
main.py — KaggleClaw entry point.

Run this one cell in a Kaggle notebook:
    import subprocess; subprocess.Popen(["python", "/kaggle/working/main.py"])

Or run directly:
    python main.py
"""

import os
import sys

# ── Load environment ───────────────────────────────────────────
# Pick up NGROK token and model config from env.py
exec(open(os.path.join(os.path.dirname(__file__), "env.py")).read())

PORT = int(os.environ.get("PORT", 8000))

# ── Start ngrok tunnel ─────────────────────────────────────────
print("🔌 Starting ngrok tunnel...")
from server.tunnel import start_tunnel
from server.state import state

try:
    public_url = start_tunnel(port=PORT, auth_token=locals().get("CONF_TOKEN") or os.environ.get("NGROK_AUTH_TOKEN"))
    state.public_url = public_url
    print(f"\n{'='*55}")
    print(f"  ⚡ KaggleClaw is LIVE")
    print(f"  🌐 Public URL: {public_url}")
    print(f"  📋 Fill in competition.md then click Start Agent")
    print(f"{'='*55}\n")
except Exception as e:
    print(f"⚠️  ngrok failed (continuing without public URL): {e}")
    public_url = f"http://localhost:{PORT}"
    print(f"   Local URL: {public_url}")

# ── Start FastAPI ──────────────────────────────────────────────
import uvicorn
from server.app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",   # keep output clean
        access_log=False,
    )

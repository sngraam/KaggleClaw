"""
server/tunnel.py — ngrok tunnel manager.
"""

import os
import time
from pyngrok import ngrok, conf


def start_tunnel(port: int = 8000, auth_token: str | None = None) -> str:
    """
    Start an ngrok HTTP tunnel on the given port.
    Returns the public HTTPS URL.
    """
    token = auth_token or os.environ.get("NGROK_AUTH_TOKEN", "")
    if token:
        ngrok.set_auth_token(token)

    # Kill any existing tunnels first
    try:
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)
    except Exception:
        pass

    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url

    # Prefer HTTPS
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://", 1)

    return public_url


def stop_tunnels():
    """Cleanly disconnect all ngrok tunnels."""
    try:
        ngrok.kill()
    except Exception:
        pass

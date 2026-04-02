"""
Cloudflare Python Worker entrypoint.

Bridges the Cloudflare Workers fetch event to FastMCP's ASGI app,
exposing the MCP Streamable HTTP transport on POST /mcp.

NOTE: Cloudflare Python Workers (Pyodide) load every .py file as a
flat top-level module. Relative imports (from .server import ...) are
not supported. All imports MUST be absolute.

Architecture:
  Cloudflare edge  -->  on_fetch()  -->  FastMCP ASGI app  -->  POST /mcp
"""

from __future__ import annotations

import os

# Set transport before importing server so FastMCP sees it at module load.
os.environ.setdefault("MCP_TRANSPORT", "streamable-http")

# Absolute import — Pyodide flat module scope, no packages.
import server  # noqa: E402

# Build the ASGI app once at cold start; reused across requests.
_app = server.mcp.http_app(path="/mcp")


async def on_fetch(request, env, ctx):
    """
    Main Worker fetch handler.

    Routes:
      GET  /     — Health check JSON
      POST /mcp  — MCP Streamable HTTP transport
      *    *      — 404
    """
    # Parse pathname from URL string
    url: str = request.url
    # Strip scheme+host: 'https://host/path?q' -> '/path'
    try:
        pathname = "/" + url.split("/", 3)[3].split("?")[0]
    except IndexError:
        pathname = "/"

    # Health / liveness probe
    if request.method == "GET" and pathname in ("/", ""):
        from cloudflare.workers import Response
        return Response(
            '{"status":"ok","transport":"streamable-http","endpoint":"/mcp"}',
            status=200,
            headers={"content-type": "application/json"},
        )

    # MCP traffic — hand off to FastMCP ASGI app
    if pathname.startswith("/mcp"):
        import asgi
        return await asgi.fetch(_app, request, env, ctx)

    from cloudflare.workers import Response
    return Response("Not found", status=404)

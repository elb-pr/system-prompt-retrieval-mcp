"""
Cloudflare Python Worker entrypoint.

Bridges the Cloudflare Workers fetch event to FastMCP's ASGI app,
exposing the MCP Streamable HTTP transport on POST /mcp.

Architecture:
  Cloudflare edge  -->  on_fetch()  -->  FastMCP ASGI app  -->  /mcp

The ASGI app is built once at module load (cold start) and reused
across requests within the same isolate lifetime.
"""

from __future__ import annotations

import os

# Cloudflare Python Workers runtime provides `js` and `asgi` builtins.
# `asgi.fetch` translates a JS Request object into an ASGI scope and
# calls the app, returning a JS Response.
from cloudflare.workers import Request, Response, fetch
import asgi

# ── Set transport before importing the server module ─────────────────────────
# The Worker always uses streamable-http. The env var may also be set
# via wrangler.jsonc vars, but we hard-code it here as a safety net.
os.environ.setdefault("MCP_TRANSPORT", "streamable-http")

# ── Build the ASGI app once at module load ────────────────────────────────────
# FastMCP.http_app() returns a Starlette ASGI application that handles:
#   POST /mcp  — Streamable HTTP MCP (the recommended transport)
#
# Import is deferred to after the env var is set so FastMCP sees it.
from .server import mcp

_app = mcp.http_app(path="/mcp")


# ── Worker entrypoint ────────────────────────────────────────────────────────

async def on_fetch(request: Request, env, ctx) -> Response:
    """
    Main fetch handler for the Cloudflare Worker.

    Routes:
      POST /mcp  — MCP Streamable HTTP transport
      GET  /     — Health check (returns 200 OK + JSON status)
      *    *      — 404
    """
    url = request.url
    pathname = url.split("?")[0].split(url.split("/")[2], 1)[-1] if "/" in url else "/"

    # Health check
    if request.method == "GET" and pathname in ("/", ""):
        return Response(
            '{"status":"ok","transport":"streamable-http","endpoint":"/mcp"}',
            status=200,
            headers={"content-type": "application/json"},
        )

    # All MCP traffic — delegate to FastMCP ASGI app
    if pathname.startswith("/mcp"):
        return await asgi.fetch(_app, request, env, ctx)

    return Response("Not found", status=404)

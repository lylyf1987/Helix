"""Shared HTTP helper for model providers."""

from __future__ import annotations

import json
import socket
import ssl
from http.client import RemoteDisconnected
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def to_runtime_error(error_prefix: str, exc: Exception) -> RuntimeError:
    """Normalize provider call failures into the runtime-facing error shape."""
    label = error_prefix.strip() or "Provider"

    if isinstance(exc, HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        return RuntimeError(f"{label} HTTP {exc.code}: {body}")

    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return RuntimeError(f"{label} invalid JSON response: {exc}")

    if isinstance(
        exc,
        (URLError, TimeoutError, socket.timeout, ConnectionError, RemoteDisconnected, ssl.SSLError),
    ):
        return RuntimeError(f"{label} network error: {exc}")

    return RuntimeError(f"{label} request failed: {exc}")


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 300,
    error_prefix: str = "Provider",
) -> dict[str, Any]:
    """POST JSON and return parsed response."""
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (
        HTTPError,
        URLError,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        RemoteDisconnected,
        ssl.SSLError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise to_runtime_error(error_prefix, exc) from exc

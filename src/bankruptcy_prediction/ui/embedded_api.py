"""Bootstrap the loopback API when Streamlit is the only managed process."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import uvicorn

from bankruptcy_prediction.api.app import create_runtime_app


@dataclass(frozen=True)
class EmbeddedApi:
    """References retained for the lifetime of the Streamlit worker."""

    server: uvicorn.Server
    thread: threading.Thread
    host: str
    port: int


_INSTANCE: EmbeddedApi | None = None
_LOCK = threading.Lock()


def _port_accepts_connections(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def ensure_embedded_api(
    *, host: str = "127.0.0.1", port: int = 8000, timeout_seconds: float = 15.0
) -> EmbeddedApi | None:
    """Start one daemonized Uvicorn server, or reuse an existing local API.

    Returning ``None`` means another process already owns the configured port;
    the Streamlit client can use that service normally.
    """

    global _INSTANCE
    with _LOCK:
        if (
            _INSTANCE is not None
            and _INSTANCE.thread.is_alive()
            and _INSTANCE.host == host
            and _INSTANCE.port == port
        ):
            return _INSTANCE
        if _port_accepts_connections(host, port):
            return None

        config = uvicorn.Config(
            create_runtime_app(),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            name="bankruptcy-embedded-api",
            daemon=True,
        )
        thread.start()
        _INSTANCE = EmbeddedApi(server=server, thread=thread, host=host, port=port)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _port_accepts_connections(host, port):
            return _INSTANCE
        if not thread.is_alive():
            break
        time.sleep(0.05)
    raise RuntimeError("Embedded FastAPI did not become ready before the timeout")

"""Wire protocol for the hydrogen host <-> client connection.

A tiny, language-agnostic framing: every message is a UTF-8 JSON object
prefixed with its byte length as a 4-byte big-endian unsigned integer::

    +----------------+------------------------+
    | uint32 length  | <length> bytes of JSON |
    +----------------+------------------------+

This module is **stdlib-only** on purpose so it can be imported by both the
client (which lives in the UI process and must stay light) and the host
(which runs under ``python -m hydrogen.service``).

Message taxonomy
----------------
Client -> host (requests) carry a correlation ``id`` so replies can be matched::

    {"id": 7, "cmd": "initialise", "args": {"system_id": "sys-1", "n": 1}}

Host -> client messages are tagged by ``type``::

    {"type": "reply",  "id": 7, "status": "ok",    "result": ...}
    {"type": "reply",  "id": 7, "status": "error", "message": "...", "kind": "..."}
    {"type": "status", "system_id": "sys-1", "phase": "running", "step": 4, ...}
    {"type": "log",    "system_id": "sys-1", "rank": 0, "level": "INFO", "message": "..."}
    {"type": "error",  "system_id": "sys-1", "kind": "NewtonConvergenceFailure", ...}
    {"type": "done",   "system_id": "sys-1"}

On-demand variable streams (opened with the ``vars_stream`` command) add a
``stream_id`` so several independent streams can be demuxed per system::

    {"type": "stream_data",   "system_id": "sys-1", "stream_id": "sys-1-stream-1",
     "initial": false, "time": [..], "series": {"src.y": [..], "lag.y": [..]}}
    {"type": "stream_closed", "system_id": "sys-1", "stream_id": "sys-1-stream-1"}

The protocol is versioned so the UI and engine can evolve independently.
"""

from __future__ import annotations

import json
import socket
import struct

PROTOCOL_VERSION = 1

_HEADER = struct.Struct("!I")
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024  # 256 MiB guard against a corrupt length


class ProtocolError(RuntimeError):
    """Raised on a malformed frame (bad length, truncated body, non-JSON)."""


def send_msg(sock: socket.socket, obj: dict) -> None:
    """Frame ``obj`` as length-prefixed JSON and send it on ``sock``.

    Callers that share a socket across threads must serialise calls to this
    function (the host routes *all* writes through a single writer thread; the
    client guards sends with a lock).
    """
    body = json.dumps(obj).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body)


def recv_msg(sock: socket.socket):
    """Read one framed message from ``sock``.

    Returns the decoded dict, or ``None`` on a clean EOF (peer closed the
    connection between messages).  Raises :class:`ProtocolError` on a truncated
    or malformed frame.
    """
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > _MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"declared message length {length} exceeds the {_MAX_MESSAGE_BYTES} "
            f"byte cap (corrupt stream?)"
        )
    body = _recv_exactly(sock, length)
    if body is None:
        raise ProtocolError("connection closed mid-message")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"could not decode message body: {exc}") from exc


def _recv_exactly(sock: socket.socket, n: int):
    """Read exactly ``n`` bytes; ``None`` on clean EOF at a frame boundary."""
    chunks = []
    remaining = n
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except (ConnectionResetError, OSError):
            if not chunks:
                return None
            raise ProtocolError("connection reset mid-message")
        if not chunk:
            if not chunks and remaining == n:
                return None  # clean EOF, nothing read yet
            raise ProtocolError("connection closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

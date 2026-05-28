from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0"


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha16(value: str) -> str:
    return sha256_hex(value)[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stage_from_kind(kind: str) -> str:
    return kind.replace("-", "_")


def evidence_envelope(
    kind: str,
    payload: dict[str, Any],
    *,
    content_policy: str = "hashed_with_samples",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a representation evidence envelope.

    The payload hash is computed over a stable JSON representation so the
    evidence record can be compared across runs. ``created_at_utc`` remains
    wall-clock metadata, but ``trace_id`` is deterministic unless the caller
    supplies an external trace identifier.
    """
    if not kind:
        raise ValueError("kind must be a non-empty string")

    canonical = stable_json(payload)
    payload_sha256 = sha256_hex(canonical)
    stage = stage_from_kind(kind)

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or f"trace_{sha16(stage + ':' + canonical)}",
        "stage": stage,
        "kind": kind,
        "content_policy": content_policy,
        "created_at_utc": utc_now_iso(),
        "payload_hash": f"sha256:{payload_sha256}",
        "payload_sha256": payload_sha256,
        "payload": payload,
        "evidence": payload,
    }

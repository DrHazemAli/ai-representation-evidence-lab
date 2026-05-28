from __future__ import annotations

from typing import Any


def to_foundry_observation(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", {})

    metrics: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ("character_count", "byte_count", "accepted_count", "candidate_count", "used_tokens"):
            if key in payload:
                metrics[key] = payload[key]

        tokenizer = payload.get("tokenizer")
        if isinstance(tokenizer, dict) and "token_count" in tokenizer:
            metrics["token_count"] = tokenizer["token_count"]

    return {
        "type": "azure_ai_foundry.representation_evidence",
        "trace_id": record.get("trace_id"),
        "stage": record.get("stage", record.get("kind")),
        "content_policy": record.get("content_policy"),
        "payload_hash": record.get("payload_hash"),
        "quality_signal": {
            "has_structured_evidence": isinstance(payload, dict),
            "schema_version": record.get("schema_version"),
        },
        "metrics": metrics,
        "evidence_ref": {
            "payload_sha256": record.get("payload_sha256"),
            "kind": record.get("kind"),
        },
    }

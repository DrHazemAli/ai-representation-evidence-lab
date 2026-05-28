from __future__ import annotations

from typing import Any


def to_otel_span(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", {})

    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "representation_evidence",
        "representation.stage": record.get("stage", record.get("kind")),
        "representation.schema_version": record.get("schema_version"),
        "representation.content_policy": record.get("content_policy"),
        "representation.payload_hash": record.get("payload_hash"),
    }

    if isinstance(payload, dict):
        if "tokenizer" in payload and isinstance(payload["tokenizer"], dict):
            tokenizer = payload["tokenizer"]
            attributes["representation.tokenizer.available"] = tokenizer.get("available")
            attributes["representation.tokenizer.name"] = tokenizer.get("encoding") or tokenizer.get("name")
            attributes["representation.token_count"] = tokenizer.get("token_count")

        for key in ("accepted_count", "candidate_count", "used_tokens", "context_sha16"):
            if key in payload:
                attributes[f"representation.context.{key}"] = payload[key]

        if record.get("stage") == "decoding_trajectory":
            for key in ("policy", "temperature", "top_p", "seed", "output_sha16"):
                if key in payload:
                    attributes[f"representation.decoding.{key}"] = payload[key]

    return {
        "name": "representation.evidence",
        "trace_id": record.get("trace_id"),
        "time_unix_nano": None,
        "attributes": attributes,
        "events": [
            {
                "name": "representation.evidence.recorded",
                "attributes": {
                    "representation.payload_sha256": record.get("payload_sha256"),
                    "representation.kind": record.get("kind"),
                },
            }
        ],
    }

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceChange:
    path: str
    before: Any
    after: Any
    message: str
    severity: str = "changed"


def read_evidence(path: str | Path) -> dict[str, Any]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("evidence file must contain a JSON object")
    return decoded


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", record.get("evidence", {}))
    return payload if isinstance(payload, dict) else {}


def _get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _append_if_changed(
    changes: list[EvidenceChange],
    before: dict[str, Any],
    after: dict[str, Any],
    path: str,
    message: str,
) -> None:
    before_value = _get(before, path)
    after_value = _get(after, path)
    if before_value != after_value:
        changes.append(EvidenceChange(path, before_value, after_value, message))


def _accepted_ids(payload: dict[str, Any]) -> list[str]:
    trace = payload.get("trace", [])
    if not isinstance(trace, list):
        return []
    return [str(item.get("id")) for item in trace if isinstance(item, dict) and item.get("decision") == "accepted"]


def _token_count(payload: dict[str, Any]) -> int | None:
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        return None
    count = tokenizer.get("token_count")
    return count if isinstance(count, int) else None


def diff_records(before: dict[str, Any], after: dict[str, Any]) -> list[EvidenceChange]:
    before_payload = _payload(before)
    after_payload = _payload(after)
    changes: list[EvidenceChange] = []

    _append_if_changed(changes, before, after, "stage", "Stage changed")
    _append_if_changed(changes, before, after, "content_policy", "Content policy changed")
    if _get(before, "payload_hash") is not None or _get(after, "payload_hash") is not None:
        _append_if_changed(changes, before, after, "payload_hash", "Payload hash changed")
    else:
        _append_if_changed(changes, before, after, "payload_sha256", "Payload hash changed")

    _append_if_changed(changes, before_payload, after_payload, "tokenizer.encoding", "Tokenizer changed")
    _append_if_changed(changes, before_payload, after_payload, "tokenizer.name", "Tokenizer changed")

    before_count = _token_count(before_payload)
    after_count = _token_count(after_payload)
    if before_count != after_count:
        changes.append(EvidenceChange("payload.tokenizer.token_count", before_count, after_count, "Token count changed"))

    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        _append_if_changed(
            changes,
            before_payload,
            after_payload,
            f"normalization.{form}.changed",
            f"{form} normalization changed",
        )
        _append_if_changed(
            changes,
            before_payload,
            after_payload,
            f"normalization.{form}.sha16",
            f"{form} normalized hash changed",
        )

    _append_if_changed(changes, before_payload, after_payload, "accepted_count", "Accepted retrieval count changed")
    _append_if_changed(changes, before_payload, after_payload, "context_sha16", "Context hash changed")
    _append_if_changed(changes, before_payload, after_payload, "used_tokens", "Context token use changed")

    before_accepted = _accepted_ids(before_payload)
    after_accepted = _accepted_ids(after_payload)
    if before_accepted != after_accepted:
        changes.append(
            EvidenceChange(
                "payload.trace.accepted_ids",
                before_accepted,
                after_accepted,
                "Accepted retrieval candidates changed",
            )
        )

    for path, message in (
        ("policy", "Decoding policy changed"),
        ("temperature", "Decoding temperature changed"),
        ("top_p", "Decoding top_p changed"),
        ("seed", "Decoding seed changed"),
        ("stop", "Decoding stop contract changed"),
        ("output_sha16", "Output hash changed"),
    ):
        _append_if_changed(changes, before_payload, after_payload, path, message)

    return _dedupe(changes)


def _dedupe(changes: list[EvidenceChange]) -> list[EvidenceChange]:
    seen: set[tuple[str, str]] = set()
    result: list[EvidenceChange] = []
    for change in changes:
        key = (change.path, change.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(change)
    return result


def format_changes(changes: list[EvidenceChange]) -> str:
    if not changes:
        return "No representation evidence changes detected."

    lines: list[str] = []
    for change in changes:
        lines.append(f"{change.message}: {change.before!r} -> {change.after!r}")
    return "\n".join(lines)


def changes_as_dicts(changes: list[EvidenceChange]) -> list[dict[str, Any]]:
    return [asdict(change) for change in changes]

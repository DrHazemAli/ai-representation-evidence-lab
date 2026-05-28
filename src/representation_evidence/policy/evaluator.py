from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..evidence import evidence_envelope, sha16, stable_json
from ..retrieval import RetrievedCandidate


@dataclass(frozen=True)
class PromotionPolicy:
    name: str
    version: int
    required: dict[str, Any]
    max_tokens: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromotionPolicy":
        required = data.get("required")
        if not isinstance(required, dict):
            raise ValueError("policy must contain a 'required' mapping")

        max_tokens = data.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("policy 'max_tokens' must be a non-negative integer")

        return cls(
            name=str(data.get("name", "context_promotion")),
            version=int(data.get("version", 1)),
            required=required,
            max_tokens=max_tokens,
        )

    @property
    def sha16(self) -> str:
        return sha16(stable_json(asdict(self)))


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to read YAML policies. Install with: pip install PyYAML") from exc

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError("policy file must decode to an object")

    return loaded


def load_policy(path: str | Path) -> PromotionPolicy:
    return PromotionPolicy.from_mapping(_load_yaml_or_json(Path(path)))


def _bool_field(data: dict[str, Any], key: str) -> bool:
    value = data[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"candidate field '{key}' must be a boolean")


def candidate_from_mapping(data: dict[str, Any]) -> RetrievedCandidate:
    try:
        return RetrievedCandidate(
            id=str(data["id"]),
            score=float(data["score"]),
            text=str(data["text"]),
            source_authority=str(data["source_authority"]),
            tenant=str(data["tenant"]),
            trusted=_bool_field(data, "trusted"),
            fresh=_bool_field(data, "fresh"),
            deleted=_bool_field(data, "deleted"),
            tokens=int(data["tokens"]),
            intended_use=str(data.get("intended_use", "answer_grounding")),
        )
    except KeyError as exc:
        raise ValueError(f"candidate is missing required field: {exc.args[0]}") from exc


def load_candidates_jsonl(path: str | Path) -> list[RetrievedCandidate]:
    candidates: list[RetrievedCandidate] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc

        if not isinstance(decoded, dict):
            raise ValueError(f"candidate line {line_number} must decode to an object")

        candidates.append(candidate_from_mapping(decoded))

    return candidates


def _allowed_authorities(policy: PromotionPolicy) -> set[str]:
    authorities = policy.required.get("allowed_authorities", [])
    if isinstance(authorities, str):
        return {authorities}
    if not isinstance(authorities, list):
        raise ValueError("policy 'allowed_authorities' must be a list or string")
    return {str(authority) for authority in authorities}


def decide_promotion_with_policy(
    candidate: RetrievedCandidate,
    *,
    tenant: str,
    used_tokens: int,
    max_tokens: int,
    policy: PromotionPolicy,
) -> str:
    required = policy.required

    if required.get("deleted") is False and candidate.deleted:
        return "rejected_deleted"

    if required.get("tenant_match", True) and candidate.tenant != tenant:
        return "rejected_wrong_tenant"

    authorities = _allowed_authorities(policy)
    if authorities and candidate.source_authority not in authorities:
        return "rejected_wrong_authority"

    if required.get("trusted_source", True) and not candidate.trusted:
        return "rejected_untrusted_source"

    if required.get("fresh", True) and not candidate.fresh:
        return "rejected_stale"

    intended_use = required.get("intended_use", "answer_grounding")
    if intended_use and candidate.intended_use != intended_use:
        return "rejected_wrong_intended_use"

    if candidate.tokens < 0:
        return "rejected_invalid_token_count"

    if used_tokens + candidate.tokens > max_tokens:
        return "rejected_token_budget"

    return "accepted"


def assemble_context_with_policy(
    candidates: Iterable[RetrievedCandidate],
    *,
    tenant: str,
    policy: PromotionPolicy,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    accepted: list[str] = []
    trace: list[dict[str, Any]] = []
    used_tokens = 0
    effective_max_tokens = policy.max_tokens if max_tokens is None else max_tokens

    if effective_max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)

    for candidate in sorted_candidates:
        decision = decide_promotion_with_policy(
            candidate,
            tenant=tenant,
            used_tokens=used_tokens,
            max_tokens=effective_max_tokens,
            policy=policy,
        )

        record = asdict(candidate)
        record.pop("text")
        record["text_sha16"] = sha16(candidate.text)
        record["decision"] = decision
        trace.append(record)

        if decision == "accepted":
            accepted.append(f"[{candidate.source_authority}:{candidate.id}]\n{candidate.text}")
            used_tokens += candidate.tokens

    context = "\n\n".join(accepted)

    payload = {
        "tenant": tenant,
        "policy": {
            "name": policy.name,
            "version": policy.version,
            "sha16": policy.sha16,
        },
        "max_tokens": effective_max_tokens,
        "used_tokens": used_tokens,
        "context_sha16": sha16(context),
        "accepted_count": len(accepted),
        "candidate_count": len(sorted_candidates),
        "trace": trace,
    }

    return context, evidence_envelope("context-promotion", payload)

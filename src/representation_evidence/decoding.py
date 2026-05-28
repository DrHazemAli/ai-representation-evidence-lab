from __future__ import annotations

import math
import random
from typing import Any

from .evidence import evidence_envelope, sha16


DEFAULT_LOGITS: list[dict[str, float]] = [
    {"APPROVE": 2.60, "REVIEW": 2.55, "DENY": 1.10},
    {"ALL": 2.20, "REFUNDS": 2.10, ".": 0.40},
    {"REFUNDS": 2.40, ".": 1.90, "</s>": 1.20},
    {".": 2.10, "</s>": 1.90},
]


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not scores:
        raise ValueError("scores must contain at least one token")

    scaled = {token: value / temperature for token, value in scores.items()}
    max_value = max(scaled.values())
    exps = {token: math.exp(value - max_value) for token, value in scaled.items()}
    total = sum(exps.values())

    return {token: value / total for token, value in exps.items()}


def greedy(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def top_p_sample(probs: dict[str, float], p: float, seed: int) -> str:
    if not (0 < p <= 1):
        raise ValueError("p must be in the interval (0, 1]")

    rng = random.Random(seed)
    items = sorted(probs.items(), key=lambda item: item[1], reverse=True)

    kept: list[tuple[str, float]] = []
    total = 0.0

    for token, prob in items:
        kept.append((token, prob))
        total += prob
        if total >= p:
            break

    threshold = rng.random()
    acc = 0.0

    for token, prob in kept:
        acc += prob / total
        if threshold <= acc:
            return token

    return kept[-1][0]


def decode_with_policy(
    logits: list[dict[str, float]] | None = None,
    policy: str = "greedy",
    temperature: float = 1.0,
    p: float = 0.8,
    seed: int = 7,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    logits = DEFAULT_LOGITS if logits is None else logits
    stop = [] if stop is None else stop

    tokens: list[str] = []
    steps: list[dict[str, Any]] = []

    for step_index, scores in enumerate(logits):
        probs = softmax(scores, temperature=temperature)

        if policy == "greedy":
            selected = greedy(probs)
        elif policy == "top_p":
            selected = top_p_sample(probs, p=p, seed=seed + step_index)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        steps.append(
            {
                "step": step_index,
                "policy": policy,
                "temperature": temperature,
                "top_p": p if policy == "top_p" else None,
                "selected_token": selected,
                "probabilities": probs,
            }
        )

        if selected == "</s>" or selected in stop:
            break

        tokens.append(selected)

    output = " ".join(tokens)

    payload = {
        "policy": policy,
        "temperature": temperature,
        "top_p": p if policy == "top_p" else None,
        "seed": seed if policy == "top_p" else None,
        "stop": stop,
        "output": output,
        "output_sha16": sha16(output),
        "steps": steps,
    }

    return evidence_envelope("decoding-trajectory", payload)


def decoding_demo() -> dict[str, Any]:
    runs = [
        decode_with_policy(policy="greedy"),
        decode_with_policy(policy="top_p", temperature=1.0, p=0.8, seed=7),
        decode_with_policy(policy="top_p", temperature=1.6, p=0.8, seed=7),
        decode_with_policy(policy="greedy", stop=["ALL"]),
    ]
    stable_runs = [
        {
            "trace_id": run["trace_id"],
            "stage": run["stage"],
            "payload_hash": run["payload_hash"],
            "evidence": run["evidence"],
        }
        for run in runs
    ]

    return evidence_envelope(
        "decoding-demo",
        {
            "description": "Same logits, different decoding contracts.",
            "runs": stable_runs,
        },
    )

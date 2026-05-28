from __future__ import annotations

import json
from pathlib import Path

from representation_evidence.decoding import decoding_demo
from representation_evidence.figures import generate_figures
from representation_evidence.policy import assemble_context_with_policy, load_policy
from representation_evidence.retrieval import assemble_context, demo_candidates
from representation_evidence.tokenizer_matrix import build_tokenizer_matrix
from representation_evidence.unicode_trace import build_input_evidence, compare_inputs


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    figures = root / "figures"

    outputs.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    input_evidence = build_input_evidence("H\u200b")
    comparison_evidence = compare_inputs("é", "e\u0301")
    context, context_evidence = assemble_context(demo_candidates(), tenant="A", max_tokens=32)
    policy = load_policy(root / "policies" / "context_promotion.yaml")
    policy_context, policy_context_evidence = assemble_context_with_policy(
        demo_candidates(),
        tenant="A",
        policy=policy,
        max_tokens=32,
    )
    tokenizer_matrix = build_tokenizer_matrix("PayPal", ["cl100k", "gpt2", "bert-base-uncased"])
    decode_evidence = decoding_demo()

    full_demo = {
        "input_evidence": input_evidence,
        "comparison_evidence": comparison_evidence,
        "final_context": context,
        "context_evidence": context_evidence,
        "policy_context": policy_context,
        "policy_context_evidence": policy_context_evidence,
        "tokenizer_matrix": tokenizer_matrix,
        "decoding_evidence": decode_evidence,
    }

    write_json(outputs / "sample-input-evidence.json", input_evidence)
    write_json(outputs / "sample-input-comparison.json", comparison_evidence)
    write_json(outputs / "sample-context-evidence.json", {"context": context, "evidence": context_evidence})
    write_json(
        outputs / "sample-policy-context-evidence.json",
        {"context": policy_context, "evidence": policy_context_evidence},
    )
    write_json(outputs / "sample-tokenizer-matrix.json", tokenizer_matrix)
    write_json(outputs / "sample-decoding-evidence.json", decode_evidence)
    write_json(outputs / "sample-full-demo.json", full_demo)

    try:
        generate_figures(figures)
    except RuntimeError as exc:
        print(f"Skipping figures: {exc}")

    print(f"Wrote outputs to: {outputs}")


if __name__ == "__main__":
    main()

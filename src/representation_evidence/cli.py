from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .decoding import decoding_demo
from .diff import changes_as_dicts, diff_records, format_changes, read_evidence
from .exporters import to_foundry_observation, to_jsonl, to_otel_span
from .figures import generate_figures
from .policy import assemble_context_with_policy, load_candidates_jsonl, load_policy
from .retrieval import assemble_context, demo_candidates
from .tokenizer_matrix import build_tokenizer_matrix
from .unicode_trace import build_input_evidence, compare_inputs


EXPORT_FORMATS = ("json", "jsonl", "otel", "otlp", "azure-foundry", "foundry")


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def emit_evidence(record: dict[str, Any], export: str) -> None:
    if export == "json":
        print_json(record)
    elif export == "jsonl":
        print(to_jsonl(record))
    elif export in {"otel", "otlp"}:
        print_json(to_otel_span(record))
    elif export in {"azure-foundry", "foundry"}:
        print_json(to_foundry_observation(record))
    else:
        raise ValueError(f"Unsupported export format: {export}")


def cmd_trace_input(args: argparse.Namespace) -> None:
    emit_evidence(build_input_evidence(args.text), args.export)


def cmd_compare(args: argparse.Namespace) -> None:
    emit_evidence(compare_inputs(args.left, args.right), args.export)


def cmd_retrieval_demo(args: argparse.Namespace) -> None:
    context, evidence = assemble_context(demo_candidates(), tenant=args.tenant, max_tokens=args.max_tokens)
    if args.export == "json":
        print_json({"final_context": context, "evidence": evidence})
    else:
        emit_evidence(evidence, args.export)


def cmd_decoding_demo(args: argparse.Namespace) -> None:
    emit_evidence(decoding_demo(), args.export)


def cmd_tokenizer_matrix(args: argparse.Namespace) -> None:
    emit_evidence(build_tokenizer_matrix(args.text, args.tokenizers), args.export)


def cmd_promote(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    candidates = load_candidates_jsonl(args.candidates)
    context, evidence = assemble_context_with_policy(
        candidates,
        tenant=args.tenant,
        policy=policy,
        max_tokens=args.max_tokens,
    )

    if args.export == "json":
        print_json({"final_context": context, "evidence": evidence})
    else:
        emit_evidence(evidence, args.export)


def cmd_diff(args: argparse.Namespace) -> None:
    changes = diff_records(read_evidence(args.before), read_evidence(args.after))
    if args.json:
        print_json(changes_as_dicts(changes))
    else:
        print(format_changes(changes))


def cmd_generate_figures(args: argparse.Namespace) -> None:
    generate_figures(Path(args.out))
    print(f"Figures generated in {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rep-evidence",
        description="Inspect representation evidence for AI systems.",
    )

    sub = parser.add_subparsers(required=True)

    trace = sub.add_parser("trace-input", help="Trace Unicode, bytes, normalization, and optional tokenizer evidence.")
    trace.add_argument("text")
    trace.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    trace.set_defaults(func=cmd_trace_input)

    compare = sub.add_parser("compare", help="Compare two inputs at representation level.")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    compare.set_defaults(func=cmd_compare)

    retrieval = sub.add_parser("retrieval-demo", help="Demonstrate retrieval-to-context promotion evidence.")
    retrieval.add_argument("--tenant", default="A")
    retrieval.add_argument("--max-tokens", type=int, default=32)
    retrieval.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    retrieval.set_defaults(func=cmd_retrieval_demo)

    matrix = sub.add_parser("tokenizer-matrix", help="Compare the same input across tokenizer contracts.")
    matrix.add_argument("text")
    matrix.add_argument("--tokenizers", nargs="+", default=["cl100k", "gpt2", "bert-base-uncased"])
    matrix.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    matrix.set_defaults(func=cmd_tokenizer_matrix)

    promote = sub.add_parser("promote", help="Promote retrieval candidates into context using a policy file.")
    promote.add_argument("--candidates", required=True)
    promote.add_argument("--policy", default="policies/context_promotion.yaml")
    promote.add_argument("--tenant", default="A")
    promote.add_argument("--max-tokens", type=int)
    promote.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    promote.set_defaults(func=cmd_promote)

    decoding = sub.add_parser("decoding-demo", help="Demonstrate decoding contract evidence.")
    decoding.add_argument("--export", choices=EXPORT_FORMATS, default="json")
    decoding.set_defaults(func=cmd_decoding_demo)

    diff = sub.add_parser("diff", help="Compare two evidence JSON records.")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.add_argument("--json", action="store_true")
    diff.set_defaults(func=cmd_diff)

    figures = sub.add_parser("generate-figures", help="Generate article figures.")
    figures.add_argument("--out", default="figures")
    figures.set_defaults(func=cmd_generate_figures)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

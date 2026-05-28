from representation_evidence.cli import build_parser


def test_cli_parser_exposes_representation_layer_commands():
    parser = build_parser()

    diff = parser.parse_args(["diff", "before.json", "after.json"])
    matrix = parser.parse_args(["tokenizer-matrix", "PayPal", "--tokenizers", "cl100k"])
    promote = parser.parse_args(["promote", "--candidates", "examples/candidates.jsonl"])

    assert diff.before == "before.json"
    assert matrix.tokenizers == ["cl100k"]
    assert promote.policy == "policies/context_promotion.yaml"

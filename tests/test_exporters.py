import json

from representation_evidence.exporters import to_foundry_observation, to_jsonl, to_otel_span
from representation_evidence.unicode_trace import build_input_evidence


def test_exporters_preserve_trace_and_hash_identity():
    evidence = build_input_evidence("H\u200b")

    jsonl = to_jsonl(evidence)
    otel = to_otel_span(evidence)
    foundry = to_foundry_observation(evidence)

    assert json.loads(jsonl)["trace_id"] == evidence["trace_id"]
    assert otel["trace_id"] == evidence["trace_id"]
    assert foundry["payload_hash"] == evidence["payload_hash"]

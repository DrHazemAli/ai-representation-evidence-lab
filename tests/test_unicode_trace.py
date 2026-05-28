from representation_evidence.unicode_trace import build_input_payload, compare_inputs


def test_composed_and_decomposed_e_acute_are_different_raw_inputs():
    left = build_input_payload("é")
    right = build_input_payload("e\u0301")

    assert left["utf8_hex"] != right["utf8_hex"]
    assert left["normalization"]["NFC"]["sha16"] == right["normalization"]["NFC"]["sha16"]


def test_zero_width_space_is_visible_in_code_points():
    payload = build_input_payload("H\u200b")
    names = [row["unicode_name"] for row in payload["code_points"]]

    assert "ZERO WIDTH SPACE" in names


def test_compare_inputs_has_evidence_envelope():
    evidence = compare_inputs("H", "h")

    assert evidence["kind"] == "input-comparison"
    assert evidence["stage"] == "input_comparison"
    assert evidence["payload_hash"].startswith("sha256:")
    assert "payload_sha256" in evidence

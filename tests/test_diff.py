from representation_evidence.diff import diff_records, format_changes
from representation_evidence.unicode_trace import build_input_evidence


def test_diff_reports_representation_changes():
    before = build_input_evidence("é")
    after = build_input_evidence("e\u0301")

    changes = diff_records(before, after)
    text = format_changes(changes)

    assert "Payload hash changed" in text
    assert "NFD normalized hash changed" not in text
    assert "NFC normalized hash changed" not in text

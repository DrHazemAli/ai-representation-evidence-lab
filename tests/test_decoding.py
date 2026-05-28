from representation_evidence.decoding import decode_with_policy, decoding_demo


def test_greedy_decoding_output():
    evidence = decode_with_policy(policy="greedy")

    assert evidence["payload"]["output"] == "APPROVE ALL REFUNDS ."


def test_stop_condition_changes_visible_output():
    evidence = decode_with_policy(policy="greedy", stop=["ALL"])

    assert evidence["payload"]["output"] == "APPROVE"


def test_decoding_demo_contains_multiple_runs():
    demo = decoding_demo()

    assert demo["kind"] == "decoding-demo"
    assert len(demo["payload"]["runs"]) == 4


def test_decoding_demo_hash_is_stable_across_timestamps():
    first = decoding_demo()
    second = decoding_demo()

    assert first["payload_hash"] == second["payload_hash"]

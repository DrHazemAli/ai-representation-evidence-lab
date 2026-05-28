from representation_evidence.tokenizer_matrix import build_tokenizer_matrix


def test_tokenizer_matrix_has_contract_rows_even_without_optional_packages():
    evidence = build_tokenizer_matrix("PayPal", ["cl100k", "bert-base-uncased"])

    assert evidence["stage"] == "tokenizer_contract_matrix"
    assert evidence["payload"]["input_summary"]["input_sha16"]
    assert [row["name"] for row in evidence["payload"]["tokenizers"]] == ["cl100k", "bert-base-uncased"]
    assert all("available" in row for row in evidence["payload"]["tokenizers"])

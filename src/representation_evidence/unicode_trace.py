from __future__ import annotations

import unicodedata
from typing import Any

from .evidence import evidence_envelope, sha16


NORMALIZATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def code_point_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, char in enumerate(text):
        rows.append(
            {
                "index": index,
                "rendered": char,
                "python_repr": repr(char),
                "code_point": f"U+{ord(char):04X}",
                "unicode_name": unicodedata.name(char, "<unassigned>"),
                "unicode_category": unicodedata.category(char),
                "utf8_hex": char.encode("utf-8").hex(" "),
            }
        )

    return rows


def normalization_rows(text: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for form in NORMALIZATION_FORMS:
        normalized = unicodedata.normalize(form, text)

        result[form] = {
            "rendered": normalized,
            "python_repr": repr(normalized),
            "sha16": sha16(normalized),
            "utf8_hex": normalized.encode("utf-8").hex(" "),
            "code_points": [f"U+{ord(char):04X}" for char in normalized],
            "changed": normalized != text,
        }

    return result


def tokenizer_rows(text: str) -> dict[str, Any]:
    """
    Optional tokenizer evidence.

    Token IDs are not universal facts. They are artifacts of a specific
    tokenizer, vocabulary, version, and model contract.
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return {
            "available": False,
            "reason": "tiktoken is not installed",
        }

    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)

    return {
        "available": True,
        "encoding": encoding_name,
        "token_ids": token_ids,
        "token_count": len(token_ids),
    }


def build_input_payload(text: str) -> dict[str, Any]:
    nfc = unicodedata.normalize("NFC", text)
    nfkc = unicodedata.normalize("NFKC", text)

    return {
        "rendered_input": text,
        "python_repr": repr(text),
        "input_sha16": sha16(text),
        "nfc_sha16": sha16(nfc),
        "nfkc_sha16": sha16(nfkc),
        "utf8_hex": text.encode("utf-8").hex(" "),
        "character_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "code_points": code_point_rows(text),
        "normalization": normalization_rows(text),
        "tokenizer": tokenizer_rows(text),
    }


def build_input_evidence(text: str) -> dict[str, Any]:
    return evidence_envelope("input-representation", build_input_payload(text))


def compare_inputs(left: str, right: str) -> dict[str, Any]:
    left_payload = build_input_payload(left)
    right_payload = build_input_payload(right)

    payload = {
        "left": left_payload,
        "right": right_payload,
        "same_python_string": left == right,
        "same_rendered": left == right,
        "same_utf8_hex": left_payload["utf8_hex"] == right_payload["utf8_hex"],
        "same_input_hash": left_payload["input_sha16"] == right_payload["input_sha16"],
        "same_nfc_hash": (
            left_payload["normalization"]["NFC"]["sha16"]
            == right_payload["normalization"]["NFC"]["sha16"]
        ),
        "same_nfkc_hash": (
            left_payload["normalization"]["NFKC"]["sha16"]
            == right_payload["normalization"]["NFKC"]["sha16"]
        ),
    }

    return evidence_envelope("input-comparison", payload)

from __future__ import annotations

import unicodedata
from typing import Any

from .evidence import evidence_envelope, sha16
from .unicode_trace import code_point_rows


TIKTOKEN_ALIASES = {
    "cl100k": "cl100k_base",
    "cl100k_base": "cl100k_base",
    "o200k": "o200k_base",
    "o200k_base": "o200k_base",
    "gpt2": "gpt2",
    "p50k": "p50k_base",
    "p50k_base": "p50k_base",
    "r50k": "r50k_base",
    "r50k_base": "r50k_base",
}


def _text_summary(text: str) -> dict[str, Any]:
    return {
        "input_sha16": sha16(text),
        "utf8_hex": text.encode("utf-8").hex(" "),
        "character_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "code_points": [row["code_point"] for row in code_point_rows(text)],
        "has_uppercase": any(char.isupper() for char in text),
        "has_non_ascii": any(ord(char) > 127 for char in text),
        "has_format_characters": any(unicodedata.category(char) == "Cf" for char in text),
    }


def _inspect_tiktoken(text: str, requested_name: str, encoding_name: str) -> dict[str, Any]:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return {
            "name": requested_name,
            "provider": "tiktoken",
            "available": False,
            "reason": "tiktoken is not installed",
        }

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        token_ids = encoding.encode(text)
        tokens = [encoding.decode([token_id]) for token_id in token_ids]
    except Exception as exc:
        return {
            "name": requested_name,
            "provider": "tiktoken",
            "encoding": encoding_name,
            "available": False,
            "reason": str(exc),
        }

    decoded = encoding.decode(token_ids)

    return {
        "name": requested_name,
        "provider": "tiktoken",
        "encoding": encoding_name,
        "available": True,
        "token_ids": token_ids,
        "tokens": tokens,
        "token_count": len(token_ids),
        "round_trip_preserved": decoded == text,
        "case_preserved": decoded == text if any(char.isupper() for char in text) else None,
        "normalization": {
            "explicit_normalizer": False,
            "changed": False,
            "note": "tiktoken encodes UTF-8 bytes directly; no tokenizer normalizer was reported.",
        },
    }


def _normalizer_output(tokenizer: Any, text: str) -> str | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    normalize_str = getattr(normalizer, "normalize_str", None)
    if callable(normalize_str):
        return str(normalize_str(text))
    return None


def _inspect_huggingface(text: str, tokenizer_name: str) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return {
            "name": tokenizer_name,
            "provider": "huggingface",
            "available": False,
            "reason": "transformers is not installed",
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
    except Exception as exc:
        return {
            "name": tokenizer_name,
            "provider": "huggingface",
            "available": False,
            "reason": str(exc),
        }

    normalized = _normalizer_output(tokenizer, text)
    decoded = tokenizer.decode(encoded, clean_up_tokenization_spaces=False)

    return {
        "name": tokenizer_name,
        "provider": "huggingface",
        "available": True,
        "tokenizer_class": tokenizer.__class__.__name__,
        "token_ids": encoded,
        "tokens": tokens,
        "token_count": len(encoded),
        "round_trip_preserved": decoded == text,
        "case_preserved": normalized == text if normalized is not None and any(char.isupper() for char in text) else None,
        "normalization": {
            "explicit_normalizer": normalized is not None,
            "changed": normalized != text if normalized is not None else None,
            "normalized_sha16": sha16(normalized) if normalized is not None else None,
            "normalized_utf8_hex": normalized.encode("utf-8").hex(" ") if normalized is not None else None,
        },
    }


def inspect_tokenizer(text: str, tokenizer_name: str) -> dict[str, Any]:
    key = tokenizer_name.lower()
    if key in TIKTOKEN_ALIASES:
        return _inspect_tiktoken(text, tokenizer_name, TIKTOKEN_ALIASES[key])
    return _inspect_huggingface(text, tokenizer_name)


def build_tokenizer_matrix(text: str, tokenizers: list[str]) -> dict[str, Any]:
    if not tokenizers:
        raise ValueError("at least one tokenizer must be requested")

    payload = {
        "input_summary": _text_summary(text),
        "rendered_input_sample": text,
        "tokenizers": [inspect_tokenizer(text, tokenizer) for tokenizer in tokenizers],
    }

    return evidence_envelope("tokenizer-contract-matrix", payload)

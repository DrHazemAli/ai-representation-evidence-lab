"""AI Representation Evidence Lab."""

from .unicode_trace import build_input_evidence, compare_inputs
from .retrieval import RetrievedCandidate, assemble_context
from .decoding import decode_with_policy
from .diff import diff_records
from .tokenizer_matrix import build_tokenizer_matrix

__all__ = [
    "build_input_evidence",
    "compare_inputs",
    "RetrievedCandidate",
    "assemble_context",
    "decode_with_policy",
    "diff_records",
    "build_tokenizer_matrix",
]

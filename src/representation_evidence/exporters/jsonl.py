from __future__ import annotations

import json
from typing import Any, Iterable


def to_jsonl(records: dict[str, Any] | Iterable[dict[str, Any]]) -> str:
    if isinstance(records, dict):
        record_iterable = [records]
    else:
        record_iterable = records

    return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in record_iterable)

from __future__ import annotations

import json
import re
from typing import Any


def extract_text(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        elif isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    combined = "".join(parts).strip()
    return combined or None


def clean_text(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    cleaned = re.sub(r"`{3}$", "", cleaned)
    return cleaned.strip()


def parse_json(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, flags=re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else trimmed
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
    return None

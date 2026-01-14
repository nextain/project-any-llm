from __future__ import annotations

import json
import re
from typing import Any


def extract_text_from_response(response: Any) -> str | None:
    parts: list[str] = []
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        if not message:
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    combined = "".join(parts).strip()
    return combined or None


def clean_text(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    cleaned = re.sub(r"`{3}$", "", cleaned)
    return cleaned.strip()


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, flags=re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else trimmed
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


def parse_scene_response(response: Any) -> str | None:
    text = extract_text_from_response(response)
    if not text:
        return None
    cleaned = clean_text(text)
    parsed = extract_json_from_text(cleaned)
    if parsed:
        scene_value = parsed.get("scene")
        if isinstance(scene_value, str) and scene_value.strip():
            return scene_value.strip()
    return cleaned if cleaned else None

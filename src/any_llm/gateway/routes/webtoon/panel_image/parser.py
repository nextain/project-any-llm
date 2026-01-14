from __future__ import annotations

import base64
import json
import re
from typing import Any, Iterable


def extract_inline_image(candidate: Any) -> tuple[str | None, str | None]:
    for part in getattr(candidate, "content", {}).get("parts", []):
        inline = part.get("inlineData")
        if inline and isinstance(inline, dict):
            data = inline.get("data")
            mime = inline.get("mimeType")
            if isinstance(data, str):
                return data, mime
    return None, None


def extract_text(candidate: Any) -> str:
    parts = getattr(candidate, "content", {}).get("parts", [])
    texts: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            texts.append(text)
    combined = "".join(texts).strip()
    return combined


def clean_text(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    cleaned = re.sub(r"`{3}$", "", cleaned)
    return cleaned.strip()


def parse_json(text: str) -> dict[str, Any] | None:
    cleaned = clean_text(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def build_metadata_summary(metadata_entries: Iterable[str | dict[str, Any]]) -> str:
    summary_lines: list[str] = []
    for entry in metadata_entries:
        if isinstance(entry, str):
            summary_lines.append(entry.strip())
        elif isinstance(entry, dict):
            for key, value in entry.items():
                summary_lines.append(f"{key}: {value}")
    return " ".join([line for line in summary_lines if line])


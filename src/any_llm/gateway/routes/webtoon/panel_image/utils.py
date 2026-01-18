"""Utility functions for panel image generation."""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image

from any_llm.gateway.log_config import logger

from .constants import (
    RESOLUTION_LONG_EDGE,
    SCENE_ELEMENT_KEYS,
    SCENE_ELEMENT_LABELS,
)
from .schema import AspectRatioType, ResolutionType


def _parse_aspect_ratio(value: AspectRatioType) -> tuple[int, int]:
    """Parse aspect ratio string into width and height."""
    parts = value.split(":")
    if len(parts) != 2:
        return (1, 1)
    try:
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            return (1, 1)
        return (width, height)
    except ValueError:
        return (1, 1)


def _resolve_target_size(
    resolution: ResolutionType, aspect_ratio: AspectRatioType
) -> tuple[int, int]:
    """Resolve target image size based on resolution and aspect ratio."""
    long_edge = RESOLUTION_LONG_EDGE.get(resolution, RESOLUTION_LONG_EDGE["1K"])
    ratio_w, ratio_h = _parse_aspect_ratio(aspect_ratio)
    if ratio_w >= ratio_h:
        width = long_edge
        height = max(1, round((long_edge * ratio_h) / ratio_w))
    else:
        width = max(1, round((long_edge * ratio_w) / ratio_h))
        height = long_edge
    return (width, height)


def _normalize_image(
    image_bytes: bytes,
    resolution: ResolutionType,
    aspect_ratio: AspectRatioType,
) -> tuple[bytes, str]:
    """Normalize image to target resolution and convert to WebP (lossless for webtoon quality)."""
    try:
        width, height = _resolve_target_size(resolution, aspect_ratio)
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            # WebP lossless: 화질 100% 유지, PNG 대비 20-30% 용량 절감
            img.save(output, format="WEBP", lossless=True)
            return output.getvalue(), "image/webp"
    except Exception as exc:
        logger.warning("Failed to normalize panel image size: %s", exc)
        return image_bytes, "image/png"


def _normalize_scene_elements(value: dict[str, str] | None) -> dict[str, str]:
    """Normalize scene elements dictionary."""
    def normalize_field(raw: str | None) -> str:
        if raw is None or raw == "null":
            return ""
        return raw

    value = value or {}
    return {
        "subject": normalize_field(value.get("subject")),
        "action": normalize_field(value.get("action")),
        "setting": normalize_field(value.get("setting")),
        "composition": normalize_field(value.get("composition")),
        "lighting": normalize_field(value.get("lighting")),
        "style": normalize_field(value.get("style")),
    }


def _has_scene_elements(elements: dict[str, str] | None) -> bool:
    """Check if scene elements contain any values."""
    if not elements:
        return False
    return any(elements.get(key, "").strip() for key in SCENE_ELEMENT_KEYS)


def _build_scene_summary(elements: dict[str, str] | None, fallback: str) -> str:
    """Build a summary string from scene elements."""
    if not elements:
        return fallback.strip()
    parts = [elements.get(key, "").strip() for key in SCENE_ELEMENT_KEYS]
    summary = " ".join([part for part in parts if part]).strip()
    return summary or fallback.strip()


def _format_scene_elements(elements: dict[str, str]) -> str:
    """Format scene elements into a readable string."""
    lines: list[str] = []
    for key in SCENE_ELEMENT_KEYS:
        value = elements.get(key, "").strip()
        if not value:
            continue
        label = SCENE_ELEMENT_LABELS.get(key, key)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _split_dialogue_lines(value: str | None) -> list[str]:
    """Split dialogue text into lines."""
    if not value:
        return []
    return [line.strip() for line in re.split(r"\r?\n", value) if line.strip()]


def _coerce_string_array(value: Any) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]\s*", value) if item.strip()]
    return []


def _extract_json_from_text(text: str) -> Any | None:
    """Extract JSON from text, handling code fences."""
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else trimmed).strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _strip_data_url(value: str) -> tuple[str | None, str | None]:
    """Strip data URL prefix and return mime type and payload."""
    if not value.startswith("data:") or "," not in value:
        return None, None
    header, payload = value.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] if header.startswith("data:") else None
    return mime_type, payload


def _parse_reference_payload(reference: str) -> tuple[str | None, str]:
    """Parse reference image payload."""
    if reference.startswith("data:") and "," in reference:
        header, payload = reference.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] if header.startswith("data:") else None
        return mime_type, payload
    return None, reference


def _extract_text_from_parts(parts: list[Any]) -> str:
    """Extract text from response parts."""
    fragments: list[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            fragments.append(text_value.strip())
    return "\n".join(fragments).strip()


def _get_response_parts(response: Any) -> list[Any]:
    """Get response parts from a Gemini response."""
    parts = getattr(response, "parts", None) or []
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        return getattr(content, "parts", None) or []
    return []


def _extract_image_and_text(parts: list[Any]) -> tuple[bytes | None, str, str]:
    """Extract image bytes, mime type, and text from response parts."""
    texts: list[str] = []
    image_bytes: bytes | None = None
    mime_type = "image/png"

    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            texts.append(text_value.strip())

        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data is not None else None
        candidate_mime = getattr(inline_data, "mime_type", None) if inline_data is not None else None
        if not data:
            continue
        if isinstance(data, bytearray):
            payload = bytes(data)
        elif isinstance(data, bytes):
            payload = data
        else:
            continue
        if isinstance(candidate_mime, str) and candidate_mime.startswith("image/"):
            mime_type = candidate_mime
        image_bytes = payload

    return image_bytes, mime_type, "\n".join(texts).strip()


def _build_character_image_parts(character_images: list[str] | None) -> list[Any]:
    """Build inline parts from character sheet images."""
    from any_llm.gateway.routes.image import _create_inline_part

    parts: list[Any] = []
    if not character_images:
        return parts
    for image_url in character_images:
        if not image_url:
            continue
        mime_type, payload = _strip_data_url(image_url)
        if not payload:
            continue
        try:
            image_bytes = base64.b64decode("".join(payload.split()), validate=True)
        except Exception:
            logger.warning("Invalid character sheet image payload, skipping")
            continue
        if not image_bytes:
            continue
        parts.append(_create_inline_part(image_bytes, mime_type or "image/png"))
    return parts


def _build_reference_parts(references: list[Any] | None) -> list[Any]:
    """Build inline parts from reference images."""
    from any_llm.gateway.routes.image import _create_inline_part

    parts: list[Any] = []
    if not references:
        return parts
    for ref in references:
        payload_raw = getattr(ref, "base64", "") or ""
        payload_raw = payload_raw.strip()
        if not payload_raw:
            continue
        inferred_mime, payload = _parse_reference_payload(payload_raw)
        mime_type = getattr(ref, "mimeType", None) or inferred_mime or "image/png"
        if not mime_type.startswith("image/"):
            mime_type = "image/png"
        try:
            image_bytes = base64.b64decode("".join(payload.split()), validate=True)
        except Exception:
            logger.warning("Invalid reference image payload, skipping")
            continue
        if not image_bytes:
            continue
        parts.append(_create_inline_part(image_bytes, mime_type))
    return parts

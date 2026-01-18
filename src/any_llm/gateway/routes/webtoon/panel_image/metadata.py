"""Metadata parsing and formatting functions for panel image generation."""
from __future__ import annotations

from typing import Any

from .utils import _coerce_string_array, _extract_json_from_text


def _parse_character_sheet_metadata(raw: Any) -> dict[str, Any] | None:
    """Parse character sheet metadata from raw input."""
    if not raw:
        return None
    payload = _extract_json_from_text(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        return None
    return {
        "summary": payload.get("summary"),
        "persona": payload.get("persona"),
        "outfit": _coerce_string_array(payload.get("outfit")),
        "colors": _coerce_string_array(payload.get("colors")),
        "accessories": _coerce_string_array(payload.get("accessories")),
        "hair": payload.get("hair"),
        "face": payload.get("face"),
        "body": payload.get("body"),
        "props": _coerce_string_array(payload.get("props")),
        "shoes": _coerce_string_array(payload.get("shoes")),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _parse_panel_metadata(raw: str | None) -> dict[str, Any] | None:
    """Parse panel metadata with 4 key elements: characters, camera, environment, continuity."""
    if not raw:
        return None
    payload = _extract_json_from_text(raw)
    if not isinstance(payload, dict):
        return None

    # Parse new 4-element structure
    characters = []
    for item in payload.get("characters") or []:
        if not isinstance(item, dict):
            continue
        characters.append(
            {
                "name": item.get("name"),
                # New position/spatial elements
                "position": item.get("position"),  # left|center|right
                "facing": item.get("facing"),  # left|right|camera
                "expression": item.get("expression"),
                # Legacy elements for backward compatibility
                "outfit": item.get("outfit"),
                "accessories": _coerce_string_array(item.get("accessories")),
                "hair": item.get("hair"),
                "props": _coerce_string_array(item.get("props")),
                "pose": item.get("pose"),
                "notes": item.get("notes"),
            }
        )

    # Parse camera info
    camera_raw = payload.get("camera") or {}
    camera = {
        "shot_type": camera_raw.get("shot_type"),  # close-up|medium|wide
        "angle": camera_raw.get("angle"),  # eye-level|low|high
    } if isinstance(camera_raw, dict) else None

    # Parse environment info
    env_raw = payload.get("environment") or {}
    environment = {
        "location": env_raw.get("location"),
        "time_of_day": env_raw.get("time_of_day"),  # morning|afternoon|evening|night
        "weather": env_raw.get("weather"),  # sunny|cloudy|rainy|snowy
        "lighting": env_raw.get("lighting"),
    } if isinstance(env_raw, dict) else None

    # Parse continuity info
    cont_raw = payload.get("continuity") or {}
    continuity = {
        "key_objects": _coerce_string_array(cont_raw.get("key_objects")),
        "spatial_notes": _coerce_string_array(cont_raw.get("spatial_notes")),
    } if isinstance(cont_raw, dict) else None

    return {
        "summary": payload.get("summary"),
        "characters": characters or None,
        "camera": camera,
        "environment": environment,
        "continuity": continuity,
        # Legacy fields for backward compatibility
        "background": payload.get("background"),
        "lighting": payload.get("lighting"),
        "changes": _coerce_string_array(payload.get("changes")),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _parse_reference_metadata(raw: str | None) -> dict[str, Any] | None:
    """Parse reference image metadata."""
    if not raw:
        return None
    payload = _extract_json_from_text(raw)
    if not isinstance(payload, dict):
        return None
    characters = []
    for item in payload.get("characters") or []:
        if not isinstance(item, dict):
            continue
        characters.append(
            {
                "name": item.get("name"),
                "outfit": item.get("outfit"),
                "accessories": _coerce_string_array(item.get("accessories")),
                "hair": item.get("hair"),
                "props": _coerce_string_array(item.get("props")),
                "pose": item.get("pose"),
                "notes": item.get("notes"),
            }
        )
    return {
        "summary": payload.get("summary"),
        "characters": characters or None,
        "background": payload.get("background"),
        "lighting": payload.get("lighting"),
        "notes": _coerce_string_array(payload.get("notes")),
    }


def _format_character_sheet_metadata(metadata: dict[str, Any]) -> str:
    """Format character sheet metadata into a readable string."""
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")
    if metadata.get("persona"):
        parts.append(f"persona: {metadata['persona']}")
    if metadata.get("outfit"):
        parts.append(f"outfit: {', '.join(metadata['outfit'])}")
    if metadata.get("colors"):
        parts.append(f"colors: {', '.join(metadata['colors'])}")
    if metadata.get("accessories"):
        parts.append(f"accessories: {', '.join(metadata['accessories'])}")
    if metadata.get("hair"):
        parts.append(f"hair: {metadata['hair']}")
    if metadata.get("face"):
        parts.append(f"face: {metadata['face']}")
    if metadata.get("body"):
        parts.append(f"body: {metadata['body']}")
    if metadata.get("props"):
        parts.append(f"props: {', '.join(metadata['props'])}")
    if metadata.get("shoes"):
        parts.append(f"shoes: {', '.join(metadata['shoes'])}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")
    return "; ".join([part for part in parts if part])


def _extract_key_visual_elements(metadata: dict[str, Any]) -> dict[str, str]:
    """Extract key visual elements that MUST remain consistent across panels."""
    return {
        "hair": metadata.get("hair") or "",
        "face": metadata.get("face") or "",
        "outfit": ", ".join(metadata.get("outfit") or []) if isinstance(metadata.get("outfit"), list) else (metadata.get("outfit") or ""),
        "colors": ", ".join(metadata.get("colors") or []) if isinstance(metadata.get("colors"), list) else (metadata.get("colors") or ""),
        "accessories": ", ".join(metadata.get("accessories") or []) if isinstance(metadata.get("accessories"), list) else (metadata.get("accessories") or ""),
    }


def _format_visual_identity_lock(name: str, elements: dict[str, str]) -> str:
    """Format key visual elements as a strict identity lock for a character."""
    lines: list[str] = []
    if elements.get("hair"):
        lines.append(f"  - HAIR (IMMUTABLE): {elements['hair']}")
    if elements.get("face"):
        lines.append(f"  - FACE (IMMUTABLE): {elements['face']}")
    if elements.get("outfit"):
        lines.append(f"  - OUTFIT (IMMUTABLE): {elements['outfit']}")
    if elements.get("colors"):
        lines.append(f"  - COLORS (IMMUTABLE): {elements['colors']}")
    if elements.get("accessories"):
        lines.append(f"  - ACCESSORIES (IMMUTABLE): {elements['accessories']}")
    if not lines:
        return ""
    return f"[{name}]\n" + "\n".join(lines)


def _format_panel_metadata(metadata: dict[str, Any]) -> str:
    """Format panel metadata with 4 key elements for continuity in subsequent panels."""
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")

    # Format characters with position/spatial info (4-element structure)
    characters = metadata.get("characters") or []
    if characters:
        character_lines = []
        for character in characters:
            name = character.get("name") or ""
            detail_parts: list[str] = []
            # New position/spatial elements (priority for continuity)
            if character.get("position"):
                detail_parts.append(f"position: {character['position']}")
            if character.get("facing"):
                detail_parts.append(f"facing: {character['facing']}")
            if character.get("expression"):
                detail_parts.append(f"expression: {character['expression']}")
            # Legacy elements
            if character.get("outfit"):
                detail_parts.append(f"outfit: {character['outfit']}")
            if character.get("accessories"):
                detail_parts.append(f"accessories: {', '.join(character['accessories'])}")
            if character.get("hair"):
                detail_parts.append(f"hair: {character['hair']}")
            if character.get("props"):
                detail_parts.append(f"props: {', '.join(character['props'])}")
            if character.get("pose"):
                detail_parts.append(f"pose: {character['pose']}")
            if character.get("notes"):
                detail_parts.append(f"notes: {character['notes']}")
            if detail_parts:
                character_lines.append(f"{name} ({', '.join(detail_parts)})")
            else:
                character_lines.append(name)
        if character_lines:
            parts.append(f"characters: {' | '.join(character_lines)}")

    # Format camera info (new 4-element structure)
    camera = metadata.get("camera")
    if camera and isinstance(camera, dict):
        camera_parts = []
        if camera.get("shot_type"):
            camera_parts.append(f"shot: {camera['shot_type']}")
        if camera.get("angle"):
            camera_parts.append(f"angle: {camera['angle']}")
        if camera_parts:
            parts.append(f"camera: {', '.join(camera_parts)}")

    # Format environment info (new 4-element structure)
    environment = metadata.get("environment")
    if environment and isinstance(environment, dict):
        env_parts = []
        if environment.get("location"):
            env_parts.append(f"location: {environment['location']}")
        if environment.get("time_of_day"):
            env_parts.append(f"time: {environment['time_of_day']}")
        if environment.get("weather"):
            env_parts.append(f"weather: {environment['weather']}")
        if environment.get("lighting"):
            env_parts.append(f"lighting: {environment['lighting']}")
        if env_parts:
            parts.append(f"environment: {', '.join(env_parts)}")

    # Format continuity info (new 4-element structure)
    continuity = metadata.get("continuity")
    if continuity and isinstance(continuity, dict):
        if continuity.get("key_objects"):
            parts.append(f"key_objects: {', '.join(continuity['key_objects'])}")
        if continuity.get("spatial_notes"):
            parts.append(f"spatial: {', '.join(continuity['spatial_notes'])}")

    # Legacy fields for backward compatibility
    if metadata.get("background"):
        parts.append(f"background: {metadata['background']}")
    if metadata.get("lighting") and not (environment and environment.get("lighting")):
        parts.append(f"lighting: {metadata['lighting']}")
    if metadata.get("changes"):
        parts.append(f"changes: {', '.join(metadata['changes'])}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")

    return "; ".join([part for part in parts if part])


def _format_reference_metadata(metadata: dict[str, Any]) -> str:
    """Format reference image metadata into a readable string."""
    parts: list[str] = []
    if metadata.get("summary"):
        parts.append(f"summary: {metadata['summary']}")
    characters = metadata.get("characters") or []
    if characters:
        character_lines = []
        for idx, character in enumerate(characters):
            name = (character.get("name") or "").strip() or f"Unknown {idx + 1}"
            detail_parts: list[str] = []
            if character.get("outfit"):
                detail_parts.append(f"outfit {character['outfit']}")
            if character.get("accessories"):
                detail_parts.append(f"accessories {', '.join(character['accessories'])}")
            if character.get("hair"):
                detail_parts.append(f"hair {character['hair']}")
            if character.get("props"):
                detail_parts.append(f"props {', '.join(character['props'])}")
            if character.get("pose"):
                detail_parts.append(f"pose {character['pose']}")
            if character.get("notes"):
                detail_parts.append(f"notes {character['notes']}")
            if detail_parts:
                character_lines.append(f"{name} ({', '.join(detail_parts)})")
            else:
                character_lines.append(name)
        if character_lines:
            parts.append(f"characters: {' | '.join(character_lines)}")
    if metadata.get("background"):
        parts.append(f"background: {metadata['background']}")
    if metadata.get("lighting"):
        parts.append(f"lighting: {metadata['lighting']}")
    if metadata.get("notes"):
        parts.append(f"notes: {', '.join(metadata['notes'])}")
    return "; ".join([part for part in parts if part])


def _build_reference_metadata_prompt() -> str:
    """Build the prompt for analyzing reference images."""
    return """
Return JSON ONLY (no markdown). Values must be in Korean.

Schema:
{
  "summary": "Common summary of the reference images",
  "characters": [
    {
      "name": "Character name or Unknown",
      "outfit": "Outfit description",
      "accessories": ["Accessories"],
      "hair": "Hair style/color",
      "props": ["Props/held items"],
      "pose": "Pose/posture",
      "notes": "Details to preserve"
    }
  ],
  "background": "Background/location",
  "lighting": "Lighting/time of day",
  "notes": ["Must-keep continuity elements"]
}

Rules:
- Only describe observable facts (no guesses).
- If multiple reference images exist, focus on shared details to preserve.
- Use empty string or empty array if unknown.
""".strip()

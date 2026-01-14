from __future__ import annotations

from typing import Iterable

ERA_IDS = ["any", "modern", "nineties", "seventies-eighties", "joseon", "future"]
SEASON_IDS = ["any", "spring", "summer", "autumn", "winter"]

ERA_LABELS = {
    "any": "Any",
    "modern": "Modern",
    "nineties": "1990s",
    "seventies-eighties": "1970s-80s",
    "joseon": "Joseon / Traditional",
    "future": "Future / Virtual",
}

SEASON_LABELS = {
    "any": "Any",
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}


def resolve_era_label(value: str | None) -> str | None:
    if not value:
        return None
    label = ERA_LABELS.get(value, "Any")
    return None if label == "Any" else label


def resolve_season_label(value: str | None) -> str | None:
    if not value:
        return None
    label = SEASON_LABELS.get(value, "Any")
    return None if label == "Any" else label


def format_scene_elements(elements: dict[str, str] | None) -> str:
    if not elements:
        return ""
    lines: list[str] = []
    for key in ("subject", "action", "setting", "composition", "lighting", "style"):
        value = elements.get(key)
        if value:
            lines.append(f"- {key.capitalize()}: {value}")
    return "\n".join(lines)


def build_world_setting_block(era_label: str | None, season_label: str | None) -> str:
    if not era_label and not season_label:
        return ""
    sections: list[str] = ["World Setting:"]
    if era_label:
        sections.append(f"- Era: {era_label}")
    if season_label:
        sections.append(f"- Season: {season_label}")
    sections.extend(
        [
            "- Make props and decorations match the era.",
            "- Treat seasonal notes as overall mood cues.",
        ]
    )
    return "\n".join(sections)


def build_scene_summary(elements: dict[str, str] | None, fallback: str) -> str:
    if not elements:
        return fallback
    parts = [elements.get(key, "").strip() for key in ("subject", "action", "setting", "composition", "lighting", "style")]
    summary = " ".join([part for part in parts if part])
    return summary or fallback


def build_prompt(
    *,
    topic: str | None,
    scene: str,
    dialogue: str | None,
    characters: Iterable[str],
    descriptions: Iterable[str] | None,
    metadata_summary: str,
    revision_note: str | None,
    scene_elements_block: str,
    world_setting_block: str,
    aspect_ratio: str,
    resolution: str,
) -> str:
    character_block = "\n".join(characters)
    description_block = "\n".join(description for description in descriptions or [] if description)
    base = f"""Scene: {scene}
Dialogue: {dialogue or "없음"}
Characters:
{character_block}
Descriptions:
{description_block or "없음"}
Metadata: {metadata_summary or "없음"}
Aspect ratio: {aspect_ratio}
Resolution: {resolution}
Revision note: {revision_note or "없음"}"""
    if scene_elements_block:
        base += f"\nScene elements:\n{scene_elements_block}"
    if world_setting_block:
        base += f"\n{world_setting_block}"
    return f"""You are producing a consistent webtoon panel illustration.
Follow all guidelines above and keep the artistic tone aligned with previous panels.
Return JSON with fields "scene" (text for metadata) and "image" (base64 inline data)."""

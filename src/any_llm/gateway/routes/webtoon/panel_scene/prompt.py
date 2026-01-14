from __future__ import annotations

from typing import Iterable

SCENE_ELEMENT_KEYS = [
    "subject",
    "action",
    "setting",
    "composition",
    "lighting",
    "style",
]

SCENE_ELEMENT_LABELS = {
    "subject": "Subject(주제)",
    "action": "Action(동작)",
    "setting": "Setting(환경)",
    "composition": "Composition(구성/카메라)",
    "lighting": "Lighting(조명)",
    "style": "Style(스타일)",
}

LANGUAGE_LABELS = {
    "ko": "한국어",
    "zh": "中文",
    "ja": "日本語",
}

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


def normalize_scene_elements(value: dict[str, str | None] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in SCENE_ELEMENT_KEYS:
        raw = (value or {}).get(key)
        normalized[key] = raw.strip() if isinstance(raw, str) else ""
    return normalized


def has_scene_elements(elements: dict[str, str]) -> bool:
    return any(elements[key].strip() for key in SCENE_ELEMENT_KEYS)


def build_scene_summary(elements: dict[str, str], fallback: str = "") -> str:
    parts = [elements[key].strip() for key in SCENE_ELEMENT_KEYS if elements[key].strip()]
    if not parts:
        return fallback.strip()
    return " ".join(parts).strip()


def format_scene_elements(elements: dict[str, str]) -> str:
    lines: list[str] = []
    for key in SCENE_ELEMENT_KEYS:
        value = elements[key].strip()
        if value:
            label = SCENE_ELEMENT_LABELS.get(key, key)
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def resolve_era_label(value: str | None) -> str | None:
    if value not in ERA_IDS:
        return None
    label = ERA_LABELS.get(value)
    return None if label == "Any" else label


def resolve_season_label(value: str | None) -> str | None:
    if value not in SEASON_IDS:
        return None
    label = SEASON_LABELS.get(value)
    return None if label == "Any" else label


def build_world_setting_block(era_label: str | None, season_label: str | None) -> str:
    sections: list[str] = []
    if not era_label and not season_label:
        return ""

    sections.append("Era/season guidance:")
    if era_label:
        sections.append(f"- Era: {era_label}")
    if season_label:
        sections.append(f"- Season: {season_label}")
    sections.extend(
        [
            "- If an era is specified, make it clearly visible in background and props.",
            "- Use season only as a mood hint.",
            "- Blend naturally without breaking the genre tone.",
        ]
    )
    return "\n".join(sections)


def build_prompt(
    *,
    scene_elements_block: str,
    fallback_scene: str,
    base_scene: str,
    dialogue: str | None,
    speaker: str | None,
    panel_number: int | None,
    topic: str | None,
    genre: str | None,
    style: str | None,
    language_label: str,
    world_setting_block: str,
) -> str:
    return f"""You are a writer refining scene descriptions for a webtoon storyboard.

Rules:
- Output must be a single JSON object.
- Use only the key "scene".
- Write naturally in {language_label}.
- Keep it concise: 1–2 sentences.
- Reflect all six elements, and naturally fill any missing elements to fit the genre/style.
- Preserve the core of the existing scene description while adding detail.
- Dialogue is for mood reference, but if it mentions props/states/actions, reflect those in the scene.
- No bullets, numbering, quotes, or parenthetical explanations.

Context:
Topic: {topic or '미정'}
Genre: {genre or '미정'}
Style: {style or '미정'}
Panel number: {panel_number if panel_number is not None else '미정'}
Speaker: {speaker or '미정'}
Dialogue: {dialogue or '미정'}
Base scene: {base_scene.strip()}
{world_setting_block + chr(10) if world_setting_block else ""}

Six elements:
{scene_elements_block or '- (없음)'}

Output format:
{{"scene":"..."}}
"""

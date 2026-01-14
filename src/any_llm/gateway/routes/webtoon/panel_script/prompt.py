from __future__ import annotations

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
    if value not in ERA_IDS:
        return None
    label = ERA_LABELS[value]
    return None if label == "Any" else label


def resolve_season_label(value: str | None) -> str | None:
    if value not in SEASON_IDS:
        return None
    label = SEASON_LABELS[value]
    return None if label == "Any" else label


def build_world_setting_block(era_label: str | None, season_label: str | None) -> str:
    if not era_label and not season_label:
        return ""
    sections = ["World Setting:"]
    if era_label:
        sections.append(f"Era: {era_label}")
    if season_label:
        sections.append(f"Season: {season_label}")
    sections.extend(
        [
            "Guidance:",
            "- If an era is provided, make it clearly visible in scene details and props.",
            "- Reinterpret modern elements into era-appropriate equivalents when needed.",
            "- Dialogue should reflect era-appropriate diction/formality without becoming hard to read.",
            "- Season should color the mood and visuals but not override the era.",
            "- If the art style implies a different era, keep the rendering style but prioritize the chosen era.",
            "- Avoid anachronisms that break the chosen era.",
        ]
    )
    return "\n".join(sections)


def build_previous_context(previous_panels: list[str]) -> str:
    if not previous_panels:
        return "없음"
    return "\n".join(previous_panels)


def build_prompt(
    *,
    topic: str,
    genre: str,
    style: str,
    world_setting_block: str,
    panel_number: int | None,
    speaker: str,
    scene: str,
    dialogue: str,
    improvement: str,
    revision_prompt: str,
    next_hint: str,
    previous_context: str,
) -> tuple[str, str]:
    system_prompt = """You are a Korean webtoon script editor.
Revise scene description and dialogue to reflect improvement feedback.
Return JSON only. No markdown."""

    user_prompt = f"""Webtoon Context:
Topic: {topic or "미정"}
Genre: {genre or "미정"}
Style: {style or "미정"}
{world_setting_block if world_setting_block else ""}

Current Panel:
Panel Number: {panel_number if panel_number is not None else "미정"}
Speaker: {speaker or "미정"}
Scene: {scene}
Dialogue: {dialogue}

Improvement Feedback:
- 개선 포인트: {improvement or "없음"}
- 수정 지시: {revision_prompt or "없음"}
- 다음 힌트: {next_hint or "없음"}

Previous Panels:
{previous_context or "없음"}

Rules:
1. Output must be valid JSON.
2. Update both scene and dialogue in Korean.
3. Keep the speaker unchanged unless absolutely necessary.
4. Keep dialogue length similar; if the input has multiple lines, preserve line count and order.
5. Each dialogue line should stay short and bubble-friendly.
6. Preserve continuity with previous panels.

Return JSON in this format:
{{
  "scene": "수정된 장면 묘사 (1~2문장)",
  "dialogue": "수정된 대사 (1~3줄)"
}}"""

    return system_prompt, user_prompt

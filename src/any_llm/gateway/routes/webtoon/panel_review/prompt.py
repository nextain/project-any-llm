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
    label = ERA_LABELS.get(value)
    return None if label == "Any" else label


def resolve_season_label(value: str | None) -> str | None:
    if value not in SEASON_IDS:
        return None
    label = SEASON_LABELS.get(value)
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
            "- If an era is provided, it should be clearly visible in props and setting.",
            "- Flag anachronisms or weak era signals in the feedback.",
        ]
    )
    return "\n".join(sections)


def build_previous_panels_block(previous_panels: list[str]) -> str:
    if not previous_panels:
        return "없음"
    return "\n".join(previous_panels)


def build_review_prompt(
    topic: str,
    genre: str,
    style: str,
    world_setting_block: str,
    panel_number: int | None,
    speaker: str,
    scene: str,
    metadata: str,
    previous_context: str,
) -> tuple[str, str]:
    system_prompt = """You are a cheerful, insightful webtoon editor and coach.
You provide short, motivating feedback in Korean that keeps creators moving to the next panel.
Return JSON only. No markdown."""

    user_prompt = f"""Webtoon Context:
Topic: {topic or '미정'}
Genre: {genre or '미정'}
Style: {style or '미정'}
{world_setting_block if world_setting_block else ''}

Current Panel:
Panel Number: {panel_number if panel_number is not None else '미정'}
Speaker: {speaker or '미정'}
Scene: {scene or '미정'}
Dialogue: (대사는 캔버스 편집으로 관리되므로 제외)
Metadata: {metadata or '없음'}

Previous Panels (for continuity):
{previous_context or '없음'}

Return JSON only in this exact structure:
{{
  "praise": "짧은 칭찬 1~2문장",
  "highlight": "특히 좋은 포인트 1문장",
  "improvement": "개선 방향 1문장",
  "nextHint": "다음 패널로 이어질 흥미로운 힌트 1문장",
  "revisionPrompt": "재생성 시 반영할 수 있는 구체 지시 1문장",
  "badge": "짧은 배지명 (예: 감정선 장인, 연출 센스)"
}}
All values must be in Korean."""

    return system_prompt, user_prompt

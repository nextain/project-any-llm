"""AI analysis functions for panel image generation."""
from __future__ import annotations

import base64
import json
from typing import Any

from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.image import _create_inline_part

from .constants import ERA_GUARDRAILS
from .metadata import (
    _build_reference_metadata_prompt,
    _format_reference_metadata,
    _parse_panel_metadata,
    _parse_reference_metadata,
)
from .utils import (
    _build_reference_parts,
    _extract_json_from_text,
    _extract_text_from_parts,
    _get_response_parts,
)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


def _analyze_reference_images(
    client: Any,
    references: list[Any],
) -> str:
    """Analyze reference images and extract metadata."""
    if not references:
        return ""
    assert genai is not None
    parts = [genai.types.Part.from_text(text=_build_reference_metadata_prompt())]
    parts.extend(_build_reference_parts(references))
    contents = [genai.types.Content(role="user", parts=parts)]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Failed to analyze reference images: %s", exc)
        return ""
    text = _extract_text_from_parts(_get_response_parts(response))
    if not text.strip():
        return ""
    parsed = _parse_reference_metadata(text)
    if parsed:
        return _format_reference_metadata(parsed)
    return text.strip()


def _analyze_era_consistency(
    client: Any,
    image_base64: str,
    mime_type: str,
    era_id: str | None,
    era_label: str | None,
    season_label: str | None,
) -> dict[str, Any]:
    """Analyze whether the generated image matches the specified era."""
    if not era_label or not image_base64:
        return {"consistent": True, "issues": [], "fix": ""}
    assert genai is not None
    guardrails = ERA_GUARDRAILS.get(era_id or "")
    must_list = guardrails.get("must", []) if guardrails else []
    avoid_list = guardrails.get("avoid", []) if guardrails else []
    prompt = f"""You are a strict visual era consistency inspector.
Check whether the image matches the specified era and report any obvious anachronisms.

Era: {era_label}
Season: {season_label or "Any"}

Must include cues: {", ".join(must_list) if must_list else "None required"}
Must avoid: {", ".join(avoid_list) if avoid_list else "None specified"}

Rules:
- If any clear anachronism is visible, set consistent=false.
- If cues are missing but no anachronisms, you may set consistent=false if the era is not visually clear.
- Keep the response concise.

Return JSON only in this format:
{{
  "consistent": true,
  "issues": ["short issue list"],
  "fix": "short correction directive for regeneration"
}}"""
    try:
        image_bytes = base64.b64decode(image_base64)
        contents = [
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(text=prompt),
                    _create_inline_part(image_bytes, mime_type),
                ],
            )
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict):
            return {
                "consistent": parsed.get("consistent", True),
                "issues": parsed.get("issues", []),
                "fix": parsed.get("fix", ""),
            }
    except Exception as exc:
        logger.warning("Era consistency check failed: %s", exc)
    return {"consistent": True, "issues": [], "fix": ""}


def _analyze_background_consistency(
    client: Any,
    image_base64: str,
    mime_type: str,
    background_base64: str,
    background_mime_type: str | None,
    scene: str | None,
    era_label: str | None,
    season_label: str | None,
) -> dict[str, Any]:
    """Analyze whether the generated image matches the background reference."""
    if not background_base64 or not image_base64:
        return {"consistent": True, "issues": [], "fix": ""}
    assert genai is not None
    prompt = f"""You are a strict background continuity inspector.
Image A is the background reference (environment baseline).
Image B is the generated panel.

Goal: confirm both images depict the same location and background elements.
Allow camera angle changes and lighting shifts, but preserve major layout, architecture, and props.

Scene: {scene or "N/A"}
Era: {era_label or "Any"}
Season: {season_label or "Any"}

Rules:
- If the location looks different or major background cues are missing, set consistent=false.
- Ignore people or character appearance; focus only on environment/props.
- Keep the response concise.

Return JSON only in this format:
{{
  "consistent": true,
  "issues": ["short issue list"],
  "fix": "short correction directive for regeneration"
}}"""
    try:
        bg_bytes = base64.b64decode(background_base64)
        img_bytes = base64.b64decode(image_base64)
        contents = [
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(text=prompt),
                    _create_inline_part(bg_bytes, background_mime_type or "image/png"),
                    _create_inline_part(img_bytes, mime_type),
                ],
            )
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict):
            return {
                "consistent": parsed.get("consistent", True),
                "issues": parsed.get("issues", []),
                "fix": parsed.get("fix", ""),
            }
    except Exception as exc:
        logger.warning("Background consistency check failed: %s", exc)
    return {"consistent": True, "issues": [], "fix": ""}


def _generate_panel_metadata(
    client: Any,
    panel_description: str,
    scene_elements_text: str,
    dialogue_lines: list[str],
    characters: list[str],
    character_descriptions: list[str] | None,
    character_sheet_metadata_lines: str,
    continuity_prompt: str,
) -> str:
    """Generate panel metadata with 4 key elements for visual continuity."""
    assert genai is not None
    prompt = f"""Return JSON ONLY (no markdown). Values must be in Korean.

Schema (4 Key Elements for Visual Continuity):
{{
  "summary": "One-sentence panel summary",
  "characters": [
    {{
      "name": "Character name",
      "position": "left|center|right (character's position in frame)",
      "facing": "left|right|camera (direction character is facing)",
      "expression": "Current facial expression",
      "outfit": "Outfit description",
      "accessories": ["Accessories"],
      "hair": "Hair style/color",
      "props": ["Props/held items"],
      "pose": "Pose/posture",
      "notes": "Details that must be preserved"
    }}
  ],
  "camera": {{
    "shot_type": "close-up|medium|wide",
    "angle": "eye-level|low|high"
  }},
  "environment": {{
    "location": "Specific location description",
    "time_of_day": "morning|afternoon|evening|night",
    "weather": "sunny|cloudy|rainy|snowy",
    "lighting": "Lighting description"
  }},
  "continuity": {{
    "key_objects": ["object: position (important objects and their positions)"],
    "spatial_notes": ["Important spatial relationships between elements"]
  }},
  "background": "Background/location (legacy)",
  "lighting": "Lighting/time of day (legacy)",
  "changes": ["Elements changed from the previous panel"],
  "notes": ["Must-keep continuity elements"]
}}

Rules:
- CRITICAL: Position and facing direction must be specified for EACH character.
- This metadata will be used to maintain visual consistency in subsequent panels.
- Only describe observable facts (no guesses).
- Use empty string or empty array if unknown.

Scene Description: {panel_description}
{f"Scene Elements:\\n{scene_elements_text}" if scene_elements_text else ""}
{f"Dialogue Cues:\\n" + "\\n".join([f"- {line}" for line in dialogue_lines]) if dialogue_lines else ""}
Characters: {", ".join(characters)}
{f"Character Details:\\n" + "\\n".join(character_descriptions) if character_descriptions else ""}
{f"Character Sheet Metadata:\\n{character_sheet_metadata_lines}" if character_sheet_metadata_lines else ""}
{continuity_prompt or ""}"""
    try:
        contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=prompt)])]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = _extract_text_from_parts(_get_response_parts(response))
        parsed = _parse_panel_metadata(text)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False)
        if text.strip():
            return json.dumps(
                {
                    "summary": text.strip(),
                    "characters": [],
                    "camera": None,
                    "environment": None,
                    "continuity": None,
                    "background": "",
                    "lighting": "",
                    "changes": [],
                    "notes": [],
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        logger.warning("Failed to generate metadata summary: %s", exc)
    return ""


def _translate_prompt(client: Any, prompt: str) -> str:
    """Translate prompt to English for image generation."""
    assert genai is not None
    if not prompt.strip():
        return prompt
    translation_prompt = (
        "You are a professional translator. Translate the following prompt into fluent English suitable for "
        "image generation without adding extra content.\n\n"
        f"{prompt}"
    )
    contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=translation_prompt)])]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Prompt translation failed: %s", exc)
        return prompt
    translated = _extract_text_from_parts(_get_response_parts(response))
    return translated.strip() or prompt


def _simplify_scene_to_keyframe(
    client: Any,
    scene_description: str,
    characters: list[str],
    dialogue: str | None,
) -> str:
    """
    Simplify a scene description to a single keyframe moment.
    Converts multi-action sequences into a single decisive moment with explicit hand positions.
    """
    assert genai is not None
    if not scene_description.strip():
        return scene_description

    character_list = ", ".join(characters) if characters else "the character"
    dialogue_context = f"\nDialogue context: {dialogue}" if dialogue else ""

    simplification_prompt = f"""You are an expert at converting complex scene descriptions into single-frame image generation prompts.

TASK: Convert the following scene description into a SINGLE DECISIVE MOMENT that can be captured in one static image.

RULES:
1. Choose only ONE moment from any action sequence (prefer the most visually interesting or emotionally significant moment)
2. Explicitly specify the position of BOTH hands for each character
3. Remove temporal words like "then", "after", "while", "as", "suddenly", "moment when"
4. Remove action transitions like "reaching for", "about to", "starting to" - show the completed state
5. If the scene describes "A하다가 B하는" (doing A then B), choose ONLY B (the final state)
6. Keep all other scene details (setting, lighting, mood, camera angle)

CHARACTER(S): {character_list}
{dialogue_context}

ORIGINAL SCENE:
{scene_description}

OUTPUT FORMAT:
Return ONLY the simplified scene description in the same language as the original. Include explicit hand positions for each character.

Example transformations:
- "진동에 놀라 휴대폰을 꺼내 확인하는" → "오른손에 휴대폰을 들고 화면을 바라보는, 왼손은 책상 위에 놓여있다"
- "커피를 마시다가 노트북을 보는" → "왼손에 커피잔을 들고 노트북 화면을 응시하는, 오른손은 키보드 위에 있다"
- "문을 열며 들어오는" → "열린 문 앞에 서있는, 오른손은 문손잡이를 잡고 있고 왼손은 가방을 들고 있다"

SIMPLIFIED SCENE:"""

    contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=simplification_prompt)])]
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    except Exception as exc:
        logger.warning("Scene simplification failed: %s", exc)
        return scene_description

    simplified = _extract_text_from_parts(_get_response_parts(response))
    return simplified.strip() or scene_description

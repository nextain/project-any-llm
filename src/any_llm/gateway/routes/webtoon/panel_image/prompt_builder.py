"""Prompt building functions for panel image generation."""
from __future__ import annotations

from .constants import (
    ANATOMICAL_NEGATIVE_PROMPT,
    ERA_GUARDRAILS,
    STYLE_PROMPTS,
)


def _get_style_prompt(style_id: str) -> dict[str, str]:
    """Get style prompt configuration by ID."""
    return STYLE_PROMPTS.get(
        style_id,
        {
            "id": "webtoon",
            "name": "웹툰 스타일",
            "systemPrompt": "당신은 웹툰 이미지 생성 AI입니다.",
            "imagePrompt": "Korean webtoon style, clean digital art",
            "negativePrompt": "low quality, blurry",
        },
    )


def _build_image_prompt(style_id: str, scene_description: str, character_descriptions: list[str]) -> str:
    """Build the full image prompt with style and scene."""
    style = _get_style_prompt(style_id)
    character_info = (
        f"\n\nCharacters in scene: {', '.join(character_descriptions)}" if character_descriptions else ""
    )
    combined_negative = f"{style['negativePrompt']}, {ANATOMICAL_NEGATIVE_PROMPT}"
    return (
        f"{style['imagePrompt']}\n\n"
        f"Scene: {scene_description}{character_info}\n\n"
        f"Style requirements: {style['systemPrompt']}\n\n"
        f"Negative prompt: {combined_negative}"
    )


def _build_era_guardrails(era_id: str | None) -> str:
    """Build era guardrails for historical consistency."""
    if not era_id:
        return ""
    guardrails = ERA_GUARDRAILS.get(era_id)
    if not guardrails:
        return ""
    lines = ["## Era Guardrails", "- The era must be visually unambiguous in the background and props."]
    if guardrails.get("must"):
        lines.append(f"- Must include cues like: {', '.join(guardrails['must'])}")
    if guardrails.get("avoid"):
        lines.append(f"- Must avoid anachronisms such as: {', '.join(guardrails['avoid'])}")
    return "\n".join(lines)


def _build_image_system_instruction(
    *,
    character_generation_mode: str | None,
    has_references: bool,
    has_character_images: bool,
    has_background_references: bool,
) -> str:
    """Build the system instruction for image generation."""
    base = (
        "CRITICAL RULES - MUST FOLLOW: "
        "1) NEVER generate ANY text, letters, numbers, symbols, captions, titles, labels, speech balloons, or written content in the image. "
        "The image must be completely text-free. This is an absolute requirement with no exceptions. "
        "2) NEVER create panel divisions, split screens, comic panels, grid layouts, or multiple frames. "
        "Generate exactly ONE single continuous scene without any borders, dividers, or panel separations. "
        "3) ANATOMICAL ACCURACY: Each character must have exactly 2 arms, 2 hands (5 fingers each), 2 legs, and 1 head. "
        "Never generate extra limbs, duplicate body parts, mutated hands, or fused fingers. "
        "4) CHARACTER VISUAL IDENTITY - ABSOLUTE REQUIREMENT: "
        "The character sheet images provided are the ONLY source of truth for character appearance. "
        "You MUST match EXACTLY: hair color, hairstyle, face shape, outfit, clothing colors, and accessories. "
        "These visual elements are IMMUTABLE and must remain identical across all panels. "
        "Do NOT modify, reinterpret, or 'improve' any visual aspect of the character. "
        "Copy the exact appearance from the character sheet image - treat it as a strict visual reference. "
        "If there is ANY conflict between the scene description and the character sheet image, "
        "ALWAYS prioritize the character sheet image for appearance (hair, face, outfit, colors, accessories). "
        "Render only one instance per named character and never duplicate the speaker. "
        "5) METADATA OUTPUT - REQUIRED: After generating the image, you MUST also output a JSON block describing the generated scene. "
        "This metadata will be used for continuity in subsequent panels. Output format: "
        '```json\\n{"characters":[{"name":"Name","position":"left|center|right","facing":"left|right|camera","expression":"expression"}],'
        '"camera":{"shot_type":"close-up|medium|wide","angle":"eye-level|low|high"},'
        '"environment":{"location":"specific location","time_of_day":"morning|afternoon|evening|night","weather":"sunny|cloudy|rainy|snowy","lighting":"description"},'
        '"continuity":{"key_objects":["object: position"],"spatial_notes":["important spatial info"]}}\\n```'
    )
    if character_generation_mode == "caricature":
        base += " Keep the output clearly stylized and avoid photorealistic rendering or likeness to real people."
    if has_references:
        base += " Reference images are PRIMARY for character appearance - match them exactly."
    if not has_references and has_character_images:
        base += " Character sheet images are PRIMARY for character appearance - match them exactly. Study the image carefully and replicate the exact visual details."
    if has_background_references:
        base += " Background reference images are environment-only; ignore any people within them and do not let them override character appearance."
    return base

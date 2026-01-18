"""Core image generation functions for panel image generation."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Coroutine

from fastapi import HTTPException, status

from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.chat import _get_model_pricing
from any_llm.gateway.routes.image import (
    _add_user_spend,
    _coerce_usage_metadata,
    _get_gemini_api_key,
    _log_image_usage,
    _set_usage_cost,
)
from any_llm.gateway.routes.utils import charge_usage_cost

from .parser import parse_json
from .prompt import resolve_era_label, resolve_season_label
from .cache import finalize_response, set_cached_panel_image
from .constants import (
    ANATOMICAL_CONSTRAINTS,
    CARICATURE_PANEL_STYLE_GUIDE,
    DEFAULT_MODEL,
)
from .analysis import (
    _analyze_background_consistency,
    _analyze_era_consistency,
    _analyze_reference_images,
    _generate_panel_metadata,
    _simplify_scene_to_keyframe,
    _translate_prompt,
)
from .metadata import (
    _extract_key_visual_elements,
    _format_character_sheet_metadata,
    _format_panel_metadata,
    _format_visual_identity_lock,
    _parse_character_sheet_metadata,
    _parse_panel_metadata,
)
from .prompt_builder import (
    _build_era_guardrails,
    _build_image_prompt,
    _build_image_system_instruction,
    _get_style_prompt,
)
from .utils import (
    _build_character_image_parts,
    _build_reference_parts,
    _build_scene_summary,
    _extract_image_and_text,
    _format_scene_elements,
    _has_scene_elements,
    _normalize_image,
    _normalize_scene_elements,
    _split_dialogue_lines,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from any_llm.gateway.config import GatewayConfig
    from any_llm.gateway.db import APIKey

    from .schema import (
        AnalysisLevelType,
        AspectRatioType,
        PanelImageResponse,
        PanelRequest,
        ResolutionType,
    )

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


ProgressCallback = Callable[[str, str], Coroutine[Any, Any, None]]


def _ensure_genai_available() -> None:
    """Ensure that the genai library is available."""
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )


async def _noop_progress(stage: str, message: str) -> None:
    """No-op progress callback."""
    pass


async def create_panel_image_response(
    payload: "PanelRequest",
    user_id: str,
    api_key: "APIKey | None",
    db: "Session",
    config: "GatewayConfig",
    cache_key: str,
    aspect_ratio: "AspectRatioType",
    resolution: "ResolutionType",
    analysis: "AnalysisLevelType",
    on_progress: ProgressCallback | None = None,
) -> "PanelImageResponse":
    """Create a panel image response with the given parameters."""
    report_progress = on_progress or _noop_progress

    provider_name = "gemini"
    model_input = DEFAULT_MODEL
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    _ensure_genai_available()
    assert genai is not None
    api_key_value = _get_gemini_api_key(config)
    client = genai.Client(api_key=api_key_value)

    await report_progress("prepare", "요청 준비 중")

    normalized_scene_elements = _normalize_scene_elements(payload.sceneElements)
    scene_value = payload.scene or ""
    scene_summary = scene_value.strip()
    if not scene_summary and _has_scene_elements(normalized_scene_elements):
        scene_summary = _build_scene_summary(normalized_scene_elements, scene_value)
    panel_description = scene_summary or scene_value
    dialogue_lines = _split_dialogue_lines(payload.dialogue)

    # Simplify scene to single keyframe to avoid multi-action rendering issues
    if analysis == "full" and panel_description.strip():
        await report_progress("keyframe", "장면 키프레임 추출 중")
        panel_description = _simplify_scene_to_keyframe(
            client,
            panel_description,
            payload.characters,
            payload.dialogue,
        )
        logger.info("Simplified scene to keyframe: %s", panel_description[:100])

    style_prompt = _get_style_prompt(payload.style)
    is_caricature = payload.characterGenerationMode == "caricature"
    character_descriptions = (
        payload.characterDescriptions
        if payload.characterDescriptions is not None
        else [f"Character: {name}" for name in payload.characters]
    )
    character_details_text = "\n".join(payload.characterDescriptions or [])
    full_image_prompt = "" if is_caricature else _build_image_prompt(
        payload.style,
        panel_description,
        character_descriptions,
    )
    role = "You are a caricature illustration expert." if is_caricature else f"You are a {style_prompt['name']} image generation expert."
    style_guide = f"{style_prompt['systemPrompt']}\n{style_prompt['imagePrompt']}"

    scene_elements_text = (
        _format_scene_elements(normalized_scene_elements) if _has_scene_elements(normalized_scene_elements) else ""
    )
    scene_elements_block = (
        "## Scene Elements (Structured)\n"
        f"{scene_elements_text}\n"
        "Use these elements to refine composition, lighting, and mood. If conflicts arise, follow the Scene Description and Dialogue Cues."
        if scene_elements_text
        else ""
    )
    dialogue_cue_block = (
        "## Dialogue Cues (Must Match Visual Details)\n"
        + "\n".join([f"- {line}" for line in dialogue_lines])
        + "\nIf the dialogue mentions props, device states, or actions (e.g., phone is off), make them visible and consistent in the scene."
        if dialogue_lines
        else ""
    )
    scene_authority_block = (
        "## Scene Authority\n"
        "- Scene Description and Dialogue Cues are authoritative for actions, props, device states, and environment.\n"
        "- Character sheets and reference images are authoritative only for character appearance (face, body, outfit).\n"
        "- If there is a conflict, keep appearance consistent but follow Scene/Dialogue for props and actions."
    )
    output_format = f"PNG ({aspect_ratio}), {resolution} quality, panel {payload.panelNumber} of 4"

    doc_instruction = ""
    if not is_caricature and payload.styleDoc and payload.styleDoc.strip():
        doc_instruction = f"Reference these style notes:\n{payload.styleDoc}\n"

    era_label = resolve_era_label(payload.era)
    season_label = resolve_season_label(payload.season)
    world_setting_block = ""
    if era_label or season_label:
        world_setting_block = (
            "## World Setting\n"
            f"{f'Era: {era_label}' if era_label else ''}\n"
            f"{f'Season: {season_label}' if season_label else ''}\n"
            "- If an era is provided, make it clearly visible through setting, props, architecture, and technology.\n"
            "- Reinterpret modern elements into era-appropriate equivalents when needed.\n"
            "- Season should color the mood, lighting, and palette without overriding the era.\n"
            "- If character sheets define outfits, keep them and emphasize the era via environment/props.\n"
            "- If the art style implies a different era, keep the rendering style but prioritize the chosen era.\n"
            "- Avoid anachronisms that break the chosen era.\n"
            "- Do not override explicit scene requirements or character sheet facts."
        ).strip()

    revision_focus = ""
    if payload.revisionNote and payload.revisionNote.strip():
        revision_focus = (
            "## Revision Focus\n"
            f"{payload.revisionNote}\n"
            "Apply the revision while preserving core composition, continuity, and character identity."
        )

    character_consistency_lines = "\n".join(
        [f"- Same character as {name}" for name in payload.characters if name]
    )
    character_sheet_metadata_lines_list: list[str] = []
    visual_identity_lock_lines: list[str] = []
    for entry in payload.characterSheetMetadata or []:
        name = entry.name.strip() if entry.name else ""
        if not name:
            continue
        parsed_metadata = _parse_character_sheet_metadata(entry.metadata)
        formatted = _format_character_sheet_metadata(parsed_metadata) if parsed_metadata else ""
        if not formatted and entry.metadata:
            formatted = str(entry.metadata).strip()
        if not formatted:
            continue
        character_sheet_metadata_lines_list.append(f"- {name}: {formatted}")
        # Extract key visual elements for identity lock
        if parsed_metadata:
            key_elements = _extract_key_visual_elements(parsed_metadata)
            identity_lock = _format_visual_identity_lock(name, key_elements)
            if identity_lock:
                visual_identity_lock_lines.append(identity_lock)
    character_sheet_metadata_lines = "\n".join(character_sheet_metadata_lines_list)
    visual_identity_lock_text = "\n\n".join(visual_identity_lock_lines)
    previous_panel_entries = payload.previousPanels or []
    continuity_notes: list[str] = []
    spatial_continuity_notes: list[str] = []  # For 4-element position/spatial info
    for previous in previous_panel_entries:
        scene_text = previous.get("scene") or ""
        dialogue_text = previous.get("dialogue") or ""
        metadata_text = previous.get("metadata") or ""
        formatted_metadata = ""
        if metadata_text:
            parsed_metadata = _parse_panel_metadata(metadata_text)
            if parsed_metadata:
                formatted_metadata = _format_panel_metadata(parsed_metadata) or metadata_text
                # Extract 4-element spatial/position info for strict continuity
                characters_meta = parsed_metadata.get("characters") or []
                for char in characters_meta:
                    char_name = char.get("name") or ""
                    position = char.get("position") or ""
                    facing = char.get("facing") or ""
                    if char_name and (position or facing):
                        spatial_continuity_notes.append(
                            f"- {char_name}: position={position}, facing={facing}"
                        )
                camera_meta = parsed_metadata.get("camera")
                if camera_meta:
                    shot = camera_meta.get("shot_type") or ""
                    angle = camera_meta.get("angle") or ""
                    if shot or angle:
                        spatial_continuity_notes.append(f"- Camera: shot={shot}, angle={angle}")
                env_meta = parsed_metadata.get("environment")
                if env_meta:
                    location = env_meta.get("location") or ""
                    if location:
                        spatial_continuity_notes.append(f"- Environment: {location}")
                cont_meta = parsed_metadata.get("continuity")
                if cont_meta:
                    key_objects = cont_meta.get("key_objects") or []
                    if key_objects:
                        spatial_continuity_notes.append(f"- Key objects: {', '.join(key_objects)}")
            else:
                formatted_metadata = metadata_text
        previous_dialogue = f" Dialogue - {dialogue_text}." if dialogue_text else ""
        continuity_notes.append(
            f"Panel {previous.get('panel')}: Scene description - {scene_text}.{previous_dialogue}"
            + (f" Metadata: {formatted_metadata}." if formatted_metadata else "")
        )
    continuity_prompt = ""
    if continuity_notes:
        spatial_block = ""
        if spatial_continuity_notes:
            spatial_block = (
                "\n\n## ⚠️ SPATIAL CONTINUITY (CRITICAL) ⚠️\n"
                "Maintain these positions and spatial relationships from the previous panel:\n"
                + "\n".join(spatial_continuity_notes)
                + "\n- Characters should stay in the SAME relative positions unless the scene explicitly describes movement."
                "\n- If a character was on the LEFT, keep them on the LEFT. If on the RIGHT, keep them on the RIGHT."
                "\n- Camera angle and shot type should remain consistent unless a scene transition occurs."
            )
        continuity_prompt = (
            "## Continuity Notes\n"
            + "\n".join(continuity_notes)
            + spatial_block
            + "\nAlways reuse the same outfit, accessories, props, and limb placements described above unless a scene explicitly calls for a change. "
            "Keep the background environment, lighting, and time-of-day consistent unless the script explicitly changes the setting. "
            "When a wardrobe, prop, or location change occurs, explain the reason while keeping facial features, hair color, and proportions consistent."
        )

    attachment_continuity_block = (
        "## Attachment Continuity Lock\n"
        "- Scope: all attached, worn, or held items (clothing layers, shoes, hats, glasses, jewelry, hair accessories, bags, belts, watches, patches, logos, patterns, handheld props, attached gadgets).\n"
        "- Rule: preserve base color, material, shape, size, and placement across panels. Do not recolor, retexture, or swap items.\n"
        "- Lighting may vary with the scene, but the item identity must remain the same.\n"
        "- Exception: if the scene explicitly changes an item, describe the change and keep all other items identical."
    )

    anatomical_accuracy_block = (
        "## Anatomical Accuracy Requirements\n"
        "- CRITICAL: Each human character must have exactly 2 arms, 2 hands (5 fingers each), 2 legs, and 1 head.\n"
        "- Never generate extra limbs, duplicate body parts, mutated hands, or fused fingers.\n"
        "- If hands are performing complex actions, simplify to a single clear gesture.\n"
        "- Avoid overlapping or intertwined limbs that may cause anatomical confusion.\n"
        "- When hands are not essential to the scene, keep them in natural resting positions or out of frame.\n"
        "- Prioritize anatomical correctness over action complexity."
    )

    pose_simplification_block = (
        "## Pose Simplification Guidelines\n"
        "- Convert complex multi-action descriptions into a single primary action.\n"
        "- If the scene describes 'doing A while doing B', focus on the most important action.\n"
        "- Specify hand positions explicitly: 'hands at sides', 'hands in pockets', 'holding [object] with both hands'.\n"
        "- Avoid foreshortening and extreme angles that may distort limb proportions.\n"
        "- Use simple, clear silhouettes that read well at a glance."
    )
    character_sheet_metadata_block = (
        "## Character Sheet Metadata (Authoritative)\n"
        f"{character_sheet_metadata_lines}\n"
        "Always follow these details exactly unless the scene explicitly requires a change.\n"
        "Reference the character sheet for exact body proportions - the character's limb count and body structure must match the reference exactly."
        if character_sheet_metadata_lines
        else ""
    )

    # Visual Identity Lock - HIGHEST PRIORITY for character appearance consistency
    visual_identity_lock_block = (
        "## ⚠️ VISUAL IDENTITY LOCK (HIGHEST PRIORITY) ⚠️\n"
        "The following visual elements are IMMUTABLE and MUST be copied exactly from the character sheet images.\n"
        "DO NOT modify, reinterpret, or change these elements under any circumstances:\n\n"
        f"{visual_identity_lock_text}\n\n"
        "ENFORCEMENT RULES:\n"
        "- Hair color and style: Copy EXACTLY from the character sheet image. No variations allowed.\n"
        "- Face shape and features: Match the reference image precisely.\n"
        "- Outfit and clothing: Replicate the exact design, colors, and patterns.\n"
        "- Accessories: Include ALL accessories shown in the character sheet.\n"
        "- If you cannot see a detail clearly in the character sheet, use the metadata description.\n"
        "- Scene descriptions may change poses and actions, but NEVER change appearance elements listed above."
        if visual_identity_lock_text
        else ""
    )

    character_anchoring_block = (
        "## Character Anchoring (Single Instance Rule)\n"
        + "\n".join([f"- Render [{name}] as a single, complete figure. Do not show {name} multiple times or from multiple angles." for name in payload.characters if name])
        + "\n- Each named character appears exactly once per panel.\n"
        "- Do not clone, mirror, or duplicate any character.\n"
        "- Background figures must be visually distinct from named characters."
        if payload.characters
        else ""
    )

    caricature_strength_lines = "\n".join([entry.strip() for entry in payload.characterCaricatureStrengths or [] if entry.strip()])
    caricature_strength_block = (
        "## Caricature Strength (Per Character)\n"
        f"{caricature_strength_lines}\n"
        "Use these levels to keep caricature exaggeration consistent across panels."
        if caricature_strength_lines
        else ""
    )

    primary_references = [ref for ref in payload.references or [] if ref.purpose not in ("background", "previous_panel")]
    background_references = [ref for ref in payload.references or [] if ref.purpose == "background"]
    previous_panel_references = [ref for ref in payload.references or [] if ref.purpose == "previous_panel"]
    has_primary_references = bool(primary_references)
    has_background_references = bool(background_references)
    has_previous_panel_reference = bool(previous_panel_references)
    has_character_images = bool(payload.characterImages)

    caricature_style_override = (
        "## Caricature Mode Override\n"
        "- Keep the overall look clearly caricatured and cartoon-like.\n"
        "- Use simplified shapes, bold outlines, and soft cel-shading.\n"
        "- Emphasize 2-3 distinctive facial features so the caricature is obvious but friendly.\n"
        "- Keep the head noticeably larger than the body and simplify limb details.\n"
        "- If the style guide conflicts with caricature exaggeration, prioritize caricature exaggeration."
        if is_caricature
        else ""
    )
    caricature_guardrails = (
        "- The output must be clearly stylized and cartoon-like.\n"
        "- Do not recreate or resemble real people, celebrities, or public figures.\n"
        "- Avoid photorealistic textures, skin detail, and lighting.\n"
        "- Exaggerate features moderately to emphasize a caricature feel while keeping it friendly."
        if is_caricature
        else ""
    )

    consistency_prompt = (
        "Character Consistency:\n"
        f"- {character_consistency_lines or 'Keep the same character.'}\n"
        + (
            "- Use the character sheet metadata below as the primary source of truth for outfits, accessories, hair, and props.\n"
            if character_sheet_metadata_lines
            else ""
        )
        + "- Keep clothing, accessories, hairstyles, and jewelry consistent with the character sheet.\n"
        "- Match the face shape, hairstyle, outfit, and props from the front-facing character sheet.\n"
        "- Preserve clothing/accessory colors, materials, logos, and placement unless the scene explicitly changes them.\n"
        "- Do not change body shape, skin tone, or hair color.\n"
        "- Maintain the same appearance so the character does not look like a different person.\n"
        "- Render only one instance per named character. Never duplicate or clone the speaker.\n"
        "- Props and device states must match the Scene Description and Dialogue Cues."
    )

    priority_rule = ""
    if has_primary_references:
        priority_rule = (
            "Priority: Reference images are primary for character appearance only. Do not let references override Scene Description "
            "or Dialogue Cues about actions, props, device states, or environment."
        )
    elif has_character_images:
        priority_rule = (
            "Priority: Character sheet images are primary for character appearance only. Do not let them override Scene Description "
            "or Dialogue Cues about actions, props, device states, or environment."
        )

    era_id = payload.era if payload.era and payload.era != "any" else None
    era_guardrails_block = _build_era_guardrails(era_id)

    should_run_analysis = analysis == "full"
    should_enforce_era = should_run_analysis and bool(era_label)
    should_enforce_background = should_run_analysis and has_background_references

    # Run metadata and reference analysis in parallel
    async def run_metadata_generation() -> str:
        if not should_run_analysis:
            return ""
        await report_progress("metadata", "패널 메타데이터 분석 중")
        return _generate_panel_metadata(
            client,
            panel_description,
            scene_elements_text,
            dialogue_lines,
            payload.characters,
            payload.characterDescriptions,
            character_sheet_metadata_lines,
            continuity_prompt,
        )

    async def run_reference_analysis() -> str:
        if not should_run_analysis or not has_primary_references:
            return ""
        await report_progress("reference-metadata", "참고 이미지 분석 중")
        return _analyze_reference_images(client, primary_references)

    metadata_summary, reference_metadata_text = await asyncio.gather(
        run_metadata_generation(),
        run_reference_analysis(),
    )
    reference_metadata_block = (
        "## Reference Image Metadata (Preserve)\n"
        f"{reference_metadata_text}\n"
        "Always preserve these visual details unless the scene explicitly changes them."
        if reference_metadata_text
        else ""
    )
    background_reference_block = (
        "## Background Reference (Environment Only)\n"
        "- Background reference images are provided to keep the location and props consistent.\n"
        "- Use them only for environment layout, architecture, and major props.\n"
        "- Do not copy any people, clothing, or character-specific details from them.\n"
        "- Keep the location consistent unless the scene explicitly changes it."
        if has_background_references
        else ""
    )

    # Previous Panel Reference - for layout and spatial consistency
    previous_panel_reference_block = (
        "## ⚠️ PREVIOUS PANEL REFERENCE (LAYOUT/COMPOSITION ONLY) ⚠️\n"
        "A previous panel image is provided to maintain visual continuity.\n"
        "USE THIS FOR:\n"
        "- Character POSITIONS (left/right placement, relative distances)\n"
        "- Camera ANGLE and PERSPECTIVE (maintain similar viewpoint)\n"
        "- Background LAYOUT (furniture, objects, architecture placement)\n"
        "- Overall COMPOSITION and FRAMING\n\n"
        "DO NOT USE THIS FOR:\n"
        "- Character APPEARANCE (hair, face, outfit) - use CHARACTER SHEET instead\n"
        "- Exact poses - follow the new scene description\n\n"
        "RULE: If characters were on the left/right in the previous panel, keep them in similar positions "
        "unless the scene explicitly describes movement or repositioning."
        if has_previous_panel_reference
        else ""
    )

    system_instruction = _build_image_system_instruction(
        character_generation_mode=payload.characterGenerationMode,
        has_references=has_primary_references,
        has_character_images=has_character_images,
        has_background_references=has_background_references,
    )

    image_config = genai.types.ImageConfig(
        aspect_ratio=aspect_ratio,
        image_size=resolution,
    )
    content_config = genai.types.GenerateContentConfig(
        response_modalities=["Text", "Image"],
        image_config=image_config,
        candidate_count=1,
    )

    def build_panel_prompt_with_corrections(
        era_correction: str = "", background_correction: str = ""
    ) -> str:
        era_correction_block = (
            f"## Era Correction\n{era_correction}\nThis correction is mandatory and must override conflicting details.\n"
            if era_correction
            else ""
        )
        background_correction_block = (
            f"## Background Correction\n{background_correction}\nThis correction is mandatory and must override conflicting details.\n"
            if background_correction
            else ""
        )
        base_text_prompt = (
            f"# Role\n{role}\n\n"
            "# Instruction\n"
            "Reference the provided **character sheet images** absolutely and depict a scene that matches the **scene description**. "
            "The character sheet images are your PRIMARY VISUAL REFERENCE - copy the appearance EXACTLY.\n"
            + (f"\n{visual_identity_lock_block}\n\n" if visual_identity_lock_block else "")
            + "---\n\n"
            "## Key Style Guide\n"
            f"{CARICATURE_PANEL_STYLE_GUIDE if is_caricature else style_guide}\n\n"
            "---\n\n"
            f"{world_setting_block}\n"
            + (f"\n{era_guardrails_block}\n" if era_guardrails_block else "")
            + era_correction_block
            + background_correction_block
            + "## Scene Information\n"
            f"Scene Description: {panel_description}\n"
            f"{scene_elements_block}\n"
            f"{dialogue_cue_block}\n"
            + (
                f"Character Details:\n{character_details_text}\n"
                if payload.characterDescriptions is not None
                else f"Characters: {', '.join(payload.characters)}\n"
            )
            + (f"\n{caricature_style_override}\n" if caricature_style_override else "")
            + (f"\n{caricature_strength_block}\n" if caricature_strength_block else "")
            + (f"\n{character_sheet_metadata_block}\n" if character_sheet_metadata_block else "")
            + (f"\n{reference_metadata_block}\n" if reference_metadata_block else "")
            + (f"\n{background_reference_block}\n" if background_reference_block else "")
            + (f"\n{previous_panel_reference_block}\n" if previous_panel_reference_block else "")
            + (f"\n{continuity_prompt}\n" if continuity_prompt else "")
            + f"\n{anatomical_accuracy_block}\n"
            + (f"\n{character_anchoring_block}\n" if character_anchoring_block else "")
            + (f"\n{revision_focus}\n" if revision_focus else "")
            + "\n## Output Format\n"
            f"{output_format}\n"
            "Additional Rule: Do not include speech balloons."
        )
        return (
            f"{full_image_prompt}\n\n{base_text_prompt}\n\n"
            f"Panel {payload.panelNumber}: {panel_description}\n"
            f"References: {', '.join(payload.characters)}\n"
            + (f"Character Details:\n{character_details_text}\n" if payload.characterDescriptions is not None else "")
            + doc_instruction
        ).strip()

    max_attempts = 1
    final_image_bytes: bytes | None = None
    final_mime_type = "image/png"
    final_text = ""
    era_correction = ""
    background_correction = ""

    for attempt in range(max_attempts):
        attempt_label = f" ({attempt + 1}/{max_attempts})" if should_enforce_era or should_enforce_background else ""

        await report_progress("translate", f"프롬프트 번역 중{attempt_label}")
        panel_prompt = build_panel_prompt_with_corrections(era_correction, background_correction)
        if should_run_analysis:
            panel_prompt = _translate_prompt(client, panel_prompt)

        prompt_with_system = f"{system_instruction}\n\n{panel_prompt}".strip()
        parts = [genai.types.Part.from_text(text=prompt_with_system)]
        parts.extend(_build_character_image_parts(payload.characterImages))
        parts.extend(_build_reference_parts(primary_references))
        parts.extend(_build_reference_parts(background_references))
        parts.extend(_build_reference_parts(previous_panel_references))
        contents: Any = [genai.types.Content(role="user", parts=parts)]

        await report_progress("generate", f"이미지 생성 중{attempt_label}")
        try:
            response = client.models.generate_content(
                model=model_input,
                contents=contents,
                config=content_config,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Panel image generation failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Panel image generation failed") from exc

        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/webtoon/generate-panel-image",
            user_id=user_id,
            usage=usage_for_charge,
        )
        if usage_for_charge:
            cost = charge_usage_cost(
                db,
                user_id=user_id,
                usage=usage_for_charge,
                model_key=model_key,
                usage_id=usage_log_id,
            )
            _set_usage_cost(db, usage_log_id, cost)
            _add_user_spend(db, user_id, cost)

        response_parts = getattr(response, "parts", None) or []
        if not response_parts:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                content = getattr(candidates[0], "content", None)
                response_parts = getattr(content, "parts", None) or []

        image_bytes, mime, text = _extract_image_and_text(response_parts)

        if not image_bytes:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned from model")

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        needs_retry = False

        # Era consistency check
        if should_enforce_era:
            await report_progress("era-check", f"시대 일치 확인 중{attempt_label}")
            era_check = _analyze_era_consistency(
                client,
                image_base64,
                mime,
                era_id,
                era_label,
                season_label,
            )
            if not era_check["consistent"] and attempt < max_attempts - 1:
                issues = era_check.get("issues", [])
                fallback_fix = (
                    f"Remove/avoid these anachronisms: {', '.join(issues)}."
                    if issues
                    else "Strengthen era-specific background and props while removing modern elements."
                )
                era_correction = (era_check.get("fix") or "").strip() or fallback_fix
                await report_progress("era-retry", "시대 불일치 감지, 재생성")
                needs_retry = True

        # Background consistency check
        if should_enforce_background:
            await report_progress("background-check", f"배경 일치 확인 중{attempt_label}")
            base_background = background_references[0] if background_references else None
            if base_background and getattr(base_background, "base64", None):
                background_check = _analyze_background_consistency(
                    client,
                    image_base64,
                    mime,
                    base_background.base64,
                    getattr(base_background, "mimeType", None),
                    panel_description,
                    era_label,
                    season_label,
                )
                if not background_check["consistent"] and attempt < max_attempts - 1:
                    issues = background_check.get("issues", [])
                    fallback_fix = (
                        f"Keep these background elements consistent: {', '.join(issues)}."
                        if issues
                        else "Match the background layout, architecture, and key props from the reference image."
                    )
                    background_correction = (background_check.get("fix") or "").strip() or fallback_fix
                    await report_progress("background-retry", "배경 불일치 감지, 재생성")
                    needs_retry = True

        if needs_retry and attempt < max_attempts - 1:
            continue

        final_image_bytes = image_bytes
        final_mime_type = mime
        final_text = text
        break

    if not final_image_bytes:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned")

    # Normalize image size
    await report_progress("normalize", "이미지 규격화 중")
    normalized_bytes, normalized_mime = _normalize_image(final_image_bytes, resolution, aspect_ratio)

    await report_progress("complete", "완료")

    # Prefer image model's actual metadata (final_text) over pre-generated metadata (metadata_summary)
    # because final_text reflects what was actually generated (positions, camera angles, etc.)
    parsed = parse_json(final_text)
    if isinstance(parsed, dict) and parsed.get("characters"):
        # Image model returned valid 4-element metadata - use it
        metadata_text = json.dumps(parsed, ensure_ascii=False)
        logger.info("Using image model's metadata with %d characters", len(parsed.get("characters", [])))
    elif final_text.strip():
        # Image model returned some text but not valid JSON - try to use it
        metadata_text = final_text.strip()
    elif metadata_summary.strip():
        # Fall back to pre-generated metadata
        metadata_text = metadata_summary.strip()
    else:
        metadata_text = ""

    inline_data = base64.b64encode(normalized_bytes).decode("utf-8")
    result = finalize_response(
        payload_text=metadata_text,
        inline_image_base64=inline_data,
        mime_type=normalized_mime,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        panel_number=payload.panelNumber,
    )
    set_cached_panel_image(cache_key, result)
    return result


async def generate_sse_stream(
    payload: "PanelRequest",
    user_id: str,
    api_key: "APIKey | None",
    db: "Session",
    config: "GatewayConfig",
    cache_key: str,
    aspect_ratio: "AspectRatioType",
    resolution: "ResolutionType",
    analysis: "AnalysisLevelType",
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for panel image generation with progress updates."""
    progress_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    result_holder: list["PanelImageResponse | Exception"] = []

    async def on_progress(stage: str, message: str) -> None:
        try:
            progress_queue.put_nowait((stage, message))
            await asyncio.sleep(0)  # Yield to event loop to allow SSE to be sent
        except Exception:
            pass

    async def run_generation() -> None:
        try:
            result = await create_panel_image_response(
                payload,
                user_id,
                api_key,
                db,
                config,
                cache_key,
                aspect_ratio,
                resolution,
                analysis,
                on_progress=on_progress,
            )
            result_holder.append(result)
        except Exception as exc:
            result_holder.append(exc)
        finally:
            await progress_queue.put(None)

    generation_task = asyncio.create_task(run_generation())

    try:
        while True:
            item = await progress_queue.get()
            if item is None:
                break
            stage, message = item
            event_data = json.dumps({"stage": stage, "message": message}, ensure_ascii=False)
            yield f"event: status\ndata: {event_data}\n\n"

        await generation_task

        if result_holder:
            result_or_error = result_holder[0]
            if isinstance(result_or_error, Exception):
                error_msg = str(result_or_error)
                error_data = json.dumps({"message": error_msg}, ensure_ascii=False)
                yield f"event: error\ndata: {error_data}\n\n"
            else:
                result_data = result_or_error.model_dump() if hasattr(result_or_error, "model_dump") else result_or_error.dict()
                yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"

        yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"
    except asyncio.CancelledError:
        generation_task.cancel()
        raise

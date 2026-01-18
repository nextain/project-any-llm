from __future__ import annotations

import base64
import io
import json
import re
from typing import Annotated, Any

from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.orm import Session  # noqa: TC002

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig  # noqa: TC001
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.chat import _get_model_pricing
from any_llm.gateway.routes.image import (
    _add_user_spend,
    _build_contents,
    _coerce_usage_metadata,
    _create_inline_part,
    _get_gemini_api_key,
    _log_image_usage,
    _set_usage_cost,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .prompt import build_prompt
from .schema import (
    CharacterSheetEntry,
    GenerateCharacterSheetRequest,
    GenerateCharacterSheetResponse,
)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_RESOLUTION = "2K"
DEFAULT_ASPECT_RATIO = "3:4"

METADATA_PROMPT = """You are a webtoon character sheet analyst. From the character sheet image below, extract the visual traits that must be preserved for consistency and return them as JSON.\n\nRules:\n- Only describe observable details (no guessing).\n- Write values in English.\n- If unknown, use an empty string or empty array.\n- Return JSON only (no code fences or explanations).\n\nSchema:\n{\n  \"summary\": \"One-sentence summary\",\n  \"persona\": \"Overall vibe/first impression\",\n  \"outfit\": [\"Top/bottom/outerwear/shoes details\"],\n  \"colors\": [\"Primary colors\"],\n  \"accessories\": [\"Accessories\"],\n  \"hair\": \"Hairstyle/color\",\n  \"face\": \"Facial features\",\n  \"body\": \"Body type/proportions\",\n  \"props\": [\"Carried items/props\"],\n  \"shoes\": [\"Shoes\"],\n  \"notes\": [\"Critical details to keep consistent\"]\n}\n"""


def _ensure_genai_available() -> None:
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )


def _extract_text_from_candidate(candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    fragments: list[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            fragments.append(text_value.strip())
    return "\n".join(fragments).strip()


def _extract_image_parts(parts: list[Any]) -> tuple[bytes | None, str, list[str], list[str]]:
    texts: list[str] = []
    thoughts: list[str] = []
    image_bytes: bytes | None = None
    mime_type = "image/png"

    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value:
            if getattr(part, "thought", False):
                thoughts.append(text_value)
            else:
                texts.append(text_value)

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
        break

    return image_bytes, mime_type, texts, thoughts


def _extract_json_from_text(text: str | None) -> Any | None:
    if not text:
        return None
    trimmed = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed, re.IGNORECASE)
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


def _get_response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None) or []
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        return getattr(content, "parts", None) or []
    return []


def _build_prompt_for_character(
    request: GenerateCharacterSheetRequest,
    character_name: str,
    character_description: str,
) -> str:
    return build_prompt(
        style_id=request.style,
        character_name=character_name,
        character_description=character_description,
        style_doc=request.styleDoc,
        era=request.era,
        season=request.season,
        scene_elements=request.sceneElements,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        resolution=DEFAULT_RESOLUTION,
    )


def _build_metadata_text(client: Any, image_bytes: bytes, mime_type: str) -> str:
    assert genai is not None
    parts = [
        genai.types.Part.from_text(text=METADATA_PROMPT),
        _create_inline_part(image_bytes, mime_type),
    ]
    contents = [genai.types.Content(role="user", parts=parts)]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai.types.GenerateContentConfig(response_modalities=["Text"], candidate_count=1),
    )
    candidate = getattr(response, "candidates", None) or []
    if not candidate:
        return ""
    return _extract_text_from_candidate(candidate[0])


def _build_data_url(mime_type: str, payload: str) -> str:
    return f"data:{mime_type};base64,{payload}"


def _convert_to_webp(image_bytes: bytes) -> tuple[bytes, str]:
    """Convert image to WebP lossless format for webtoon quality."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA")
            output = io.BytesIO()
            img.save(output, format="WEBP", lossless=True)
            return output.getvalue(), "image/webp"
    except Exception as exc:
        logger.warning("Failed to convert character sheet to WebP: %s", exc)
        return image_bytes, "image/png"


@router.post("/character-sheet", response_model=GenerateCharacterSheetResponse)
async def generate_character_sheet(
    http_request: Request,
    auth_result: Annotated[
        tuple[APIKey | None, bool, str | None, SessionToken | None],
        Depends(verify_jwt_or_api_key_or_master),
    ],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GenerateCharacterSheetResponse:
    try:
        payload = await http_request.json()
        logger.warning("Character sheet request body: %s", payload)
    except Exception as exc:
        logger.warning("Failed to read character sheet request body: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body") from exc

    try:
        request = GenerateCharacterSheetRequest.model_validate(payload)
    except ValidationError as exc:
        logger.warning(
            "Character sheet validation failed: %s %s",
            http_request.url.path,
            {
                "errors": exc.errors(),
                "body": payload,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Invalid character sheet payload",
                "errors": exc.errors(),
                "body": payload,
            },
        ) from exc
    api_key, _, _, _ = auth_result
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' detail is required",
    )
    validate_user_credit(db, user_id)

    model_input = request.model or DEFAULT_MODEL
    provider_name = "gemini"
    model = model_input
    model_key, _ = _get_model_pricing(db, provider_name, model)

    _ensure_genai_available()
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )
    api_key_value = _get_gemini_api_key(config)
    client = genai.Client(api_key=api_key_value)

    image_config = genai.types.ImageConfig(
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        image_size=DEFAULT_RESOLUTION,
    )
    content_config = genai.types.GenerateContentConfig(
        response_modalities=["Text", "Image"],
        image_config=image_config,
        candidate_count=1,
        thinking_config=genai.types.ThinkingConfig(include_thoughts=True),
    )

    character_sheets: list[CharacterSheetEntry] = []

    for character in request.characters:
        prompt_text = _build_prompt_for_character(
            request,
            character.name,
            character.description,
        )
        contents: Any = _build_contents(prompt_text, None)

        try:
            response = client.models.generate_content(
                model=model_input,
                contents=contents,
                config=content_config,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("character sheet generation failed for %s: %s", character.name, exc)
            character_sheets.append(
                CharacterSheetEntry(name=character.name, imageUrl=None, metadata=""),
            )
            continue

        parts = _get_response_parts(response)
        image_bytes, mime_type, _, _ = _extract_image_parts(parts)
        usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
        usage_log_id = _log_image_usage(
            db=db,
            api_key_obj=api_key,
            model=model_input,
            provider=provider_name,
            endpoint="/v1/webtoon/character-sheet",
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

        metadata_text = ""
        image_url = None
        if image_bytes:
            metadata_text = _build_metadata_text(client, image_bytes, mime_type)
            parsed_metadata = _extract_json_from_text(metadata_text)
            metadata_text = json.dumps(parsed_metadata) if isinstance(parsed_metadata, dict) else (metadata_text or "")
            webp_bytes, webp_mime = _convert_to_webp(image_bytes)
            base64_payload = base64.b64encode(webp_bytes).decode("utf-8")
            image_url = _build_data_url(webp_mime, base64_payload)
        else:
            logger.error("character sheet returned no image parts for %s", character.name)

        character_sheets.append(
            CharacterSheetEntry(name=character.name, imageUrl=image_url, metadata=metadata_text),
        )

    return GenerateCharacterSheetResponse(characterSheets=character_sheets)

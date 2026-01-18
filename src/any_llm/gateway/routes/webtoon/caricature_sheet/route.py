from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Annotated, Any, TYPE_CHECKING

from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from any_llm import AnyLLM
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
    CaricatureSheetEntry,
    GenerateCaricatureSheetRequest,
    GenerateCaricatureSheetResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_RESOLUTION = "2K"
DEFAULT_ASPECT_RATIO = "3:4"

METADATA_PROMPT = """You are a webtoon caricature sheet analyst. From the caricature image below, summarize the identifiable visual traits that should stay consistent. Return JSON only with these fields: summary, persona, outfit, colors, accessories, hair, face, body, props, shoes, notes."""


def _ensure_genai_available() -> None:
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai dependency is not installed",
        )


def _parse_data_url(value: str) -> tuple[str, str]:
    match = re.match(r"^data:(.*?);base64,(.*)$", value)
    if not match:
        raise ValueError("Invalid data URL")
    mime_type, payload = match.groups()
    return mime_type, payload


def _fetch_image(reference: str) -> tuple[str, str]:
    import httpx

    if reference.startswith("data:"):
        return _parse_data_url(reference)
    response = httpx.get(reference, timeout=30.0)
    response.raise_for_status()
    mime_type = response.headers.get("content-type", "image/png")
    payload = base64.b64encode(response.content).decode("utf-8")
    return mime_type, payload


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


def _extract_text(candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    fragments: list[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            fragments.append(text_value.strip())
    return "\n".join(fragments).strip()


def _get_response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None) or []
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        return getattr(content, "parts", None) or []
    return []


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
    return _extract_text(candidate[0])


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
        logger.warning("Failed to convert caricature to WebP: %s", exc)
        return image_bytes, "image/png"


@router.post("/caricature-sheet", response_model=GenerateCaricatureSheetResponse)
async def generate_caricature_sheet(
    http_request: Request,
    auth_result: Annotated[
        tuple[APIKey | None, bool, str | None, SessionToken | None],
        Depends(verify_jwt_or_api_key_or_master),
    ],
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> GenerateCaricatureSheetResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    try:
        payload = await http_request.json()
        logger.info("Caricature sheet request body: %s", payload)
    except Exception as exc:
        logger.warning("Failed to read caricature request body: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body") from exc

    try:
        request = GenerateCaricatureSheetRequest.model_validate(payload)
    except ValidationError as exc:
        logger.warning(
            "Caricature sheet payload invalid: %s %s",
            http_request.url.path,
            {
                "errors": exc.errors(),
                "body": payload,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Invalid caricature payload",
                "errors": exc.errors(),
                "body": payload,
            },
        ) from exc

    mime_type, payload_data = await asyncio.to_thread(_fetch_image, request.referenceImage)
    model_input = DEFAULT_MODEL
    provider_name = "gemini"
    model_key, _ = _get_model_pricing(db, provider_name, model_input)

    _ensure_genai_available()
    assert genai is not None
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

    prompt_text = build_prompt(request, mime_type)
    contents: Any = _build_contents(prompt_text, [f"data:{mime_type};base64,{payload_data}"])

    try:
        response = client.models.generate_content(
            model=model_input,
            contents=contents,
            config=content_config,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Caricature generation failed for %s: %s", request.name, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Caricature generation failed") from exc

    parts = _get_response_parts(response)
    image_bytes, result_mime, _, _ = _extract_image_parts(parts)
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned")

    usage_info = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    usage_for_charge = _coerce_usage_metadata(usage_info) or usage_info
    usage_log_id = _log_image_usage(
        db=db,
        api_key_obj=api_key,
        model=model_input,
        provider=provider_name,
        endpoint="/v1/webtoon/caricature-sheet",
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

    metadata_text = _build_metadata_text(client, image_bytes, result_mime)
    parsed_metadata = {}
    if metadata_text:
        sanitized = metadata_text.strip()
        sanitized = re.sub(r"^```(?:json)?\s*", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s*```$", "", sanitized)
        try:
            parsed_metadata = json.loads(sanitized)
        except json.JSONDecodeError:
            logger.warning("caricature metadata is not valid json: %s", sanitized)
    webp_bytes, webp_mime = _convert_to_webp(image_bytes)
    base64_payload = base64.b64encode(webp_bytes).decode("utf-8")
    image_url = _build_data_url(webp_mime, base64_payload)

    return GenerateCaricatureSheetResponse(
        sheet=CaricatureSheetEntry(imageUrl=image_url, metadata=json.dumps(parsed_metadata)),
    )

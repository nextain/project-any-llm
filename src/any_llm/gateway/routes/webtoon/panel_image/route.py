from __future__ import annotations

import json
import time
import hashlib
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from any_llm import AnyLLM, acompletion
from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.chat import (
    _get_model_pricing,
    _get_provider_kwargs,
    _log_usage,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .parser import extract_inline_image, extract_text, build_metadata_summary, parse_json
from .prompt import (
    build_prompt,
    build_world_setting_block,
    format_scene_elements,
    resolve_era_label,
    resolve_season_label,
)
from .schema import (
    PanelImageResponse,
    PanelRequest,
    StatusUpdate,
)

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])

IMAGE_CACHE: dict[str, tuple[PanelImageResponse, float]] = {}

MODEL_INPUT = "gemini:gemini-3-pro-image-preview"
DEFAULT_RESOLUTION = "1K"
DEFAULT_ASPECT_RATIO = "1:1"


def build_cache_key(body: PanelRequest, aspect_ratio: str, resolution: str, analysis_level: str) -> str:
    hash_input = (
        f"{body.scene}|{body.dialogue or ''}|{','.join(body.characters)}|{body.style}|{body.panelNumber}|"
        f"{body.era or ''}|{body.season or ''}|{aspect_ratio}|{resolution}|{analysis_level}|{body.revisionNote or ''}"
    )
    for ref in body.references or []:
        hash_input += f"|ref-{ref.base64}-{ref.mimeType or ''}-{ref.purpose or ''}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def get_cached_panel_image(key: str) -> PanelImageResponse | None:
    entry = IMAGE_CACHE.get(key)
    if not entry:
        return None
    payload, expires = entry
    if expires < time.time():
        del IMAGE_CACHE[key]
        return None
    return payload


def set_cached_panel_image(key: str, payload: PanelImageResponse):
    IMAGE_CACHE[key] = (payload, time.time() + 5 * 60)


def finalize_response(
    payload_text: str,
    inline_image_base64: str,
    mime_type: str,
    aspect_ratio: str,
    resolution: str,
    panel_number: int,
) -> PanelImageResponse:
    return PanelImageResponse(
        success=True,
        imageUrl=f"data:{mime_type};base64,{inline_image_base64}",
        imageBase64=inline_image_base64,
        mimeType=mime_type,
        metadata=payload_text,
        text=payload_text,
        aspectRatio=aspect_ratio,
        resolution=resolution,
        model="gemini-3-pro-image-preview",
        panelNumber=panel_number,
    )


STATUS_STEPS = [
    StatusUpdate(stage="prepare", message="패널 정보 준비 중"),
    StatusUpdate(stage="prompt", message="프롬프트 생성 중"),
    StatusUpdate(stage="ai", message="AI 세션 진행 중"),
    StatusUpdate(stage="finalize", message="이미지 정리 중"),
]


def make_event(event: str, payload: Any) -> str:
    safe_data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {safe_data}\n\n"


async def create_panel_image_response(
    payload: PanelRequest,
    user_id: str,
    api_key: APIKey | None,
    db: Session,
    config: GatewayConfig,
    cache_key: str,
    aspect_ratio: str,
    resolution: str,
    analysis: str,
) -> PanelImageResponse:
    era = resolve_era_label(payload.era)
    season = resolve_season_label(payload.season)
    scene_elements_block = format_scene_elements(payload.sceneElements or {})
    world_setting_block = build_world_setting_block(era, season)
    metadata_entries: list[Any] = []
    for entry in payload.characterSheetMetadata or []:
        if entry.metadata:
            metadata_entries.append(entry.metadata)
    metadata_summary = build_metadata_summary(metadata_entries)

    prompt = build_prompt(
        topic=None,
        scene=payload.scene,
        dialogue=payload.dialogue,
        characters=payload.characters,
        descriptions=payload.characterDescriptions,
        metadata_summary=metadata_summary,
        revision_note=payload.revisionNote,
        scene_elements_block=scene_elements_block,
        world_setting_block=world_setting_block,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    provider, model = AnyLLM.split_model_provider(MODEL_INPUT)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_kwargs(config, provider)

    completion_kwargs = {
        "model": MODEL_INPUT,
        "contents": [{"text": f"{prompt}\nAnalysis level: {analysis}"}],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    response = await acompletion(**completion_kwargs)
    usage_log_id = await _log_usage(
        db=db,
        api_key_obj=api_key,
        model=model,
        provider=provider,
        endpoint="/v1/webtoon/generate-panel-image",
        user_id=user_id,
        response=response,
        model_key=model_key,
        model_pricing=model_pricing,
    )
    charge_usage_cost(
        db,
        user_id=user_id,
        usage=getattr(response, "usage", None),
        model_key=model_key,
        usage_id=usage_log_id,
    )
    text = extract_text(response)
    parsed = parse_json(text)
    if not parsed:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Invalid AI response")
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned")
    inline_data, mime = extract_inline_image(candidates[0])
    if not inline_data or not mime:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No image returned")
    result = finalize_response(
        payload_text=text,
        inline_image_base64=inline_data,
        mime_type=mime,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        panel_number=payload.panelNumber,
    )
    set_cached_panel_image(cache_key, result)
    return result


async def panel_image_event_stream(
    payload: PanelRequest,
    user_id: str,
    api_key: APIKey | None,
    db: Session,
    config: GatewayConfig,
    cache_key: str,
    aspect_ratio: str,
    resolution: str,
    analysis: str,
) -> AsyncGenerator[str, None]:
    for step in STATUS_STEPS:
        yield make_event("status", step.dict())
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
        )
    except HTTPException as exc:
        yield make_event("error", {"message": exc.detail or "Failed to generate"})
        return
    except Exception as exc:
        logger.error("Panel image SSE stream failed: %s", exc)
        yield make_event("error", {"message": str(exc)})
        return
    yield make_event("result", result.dict())
    yield make_event("done", {"ok": True})


@router.post("/generate-panel-image", response_model=PanelImageResponse)
async def generate_panel_image(
    request: Request,
    payload: PanelRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
):
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    accept_header = request.headers.get("accept", "")
    wants_stream = "text/event-stream" in accept_header.lower()

    aspect_ratio = payload.aspectRatio or "1:1"
    resolution = payload.resolution or "1K"
    analysis = payload.analysisLevel or "fast"
    cache_key = build_cache_key(payload, aspect_ratio, resolution, analysis)
    cached = get_cached_panel_image(cache_key)
    if cached:
        if wants_stream:
            payload_data = make_event("result", cached.dict()) + make_event("done", {"ok": True})
            return StreamingResponse(
                iter([payload_data]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                },
            )
        return cached

    if wants_stream:
        return StreamingResponse(
            panel_image_event_stream(
                payload,
                user_id,
                api_key,
                db,
                config,
                cache_key,
                aspect_ratio,
                resolution,
                analysis,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
            },
        )

    return await create_panel_image_response(
        payload,
        user_id,
        api_key,
        db,
        config,
        cache_key,
        aspect_ratio,
        resolution,
        analysis,
    )

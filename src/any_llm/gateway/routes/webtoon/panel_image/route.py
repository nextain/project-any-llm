"""Panel image generation route handler."""
from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.utils import resolve_target_user, validate_user_credit

from .cache import build_cache_key, get_cached_panel_image
from .constants import DEFAULT_ASPECT_RATIO, DEFAULT_RESOLUTION
from .generator import create_panel_image_response, generate_sse_stream
from .schema import (
    AnalysisLevelType,
    AspectRatioType,
    PanelImageResponse,
    PanelRequest,
    ResolutionType,
)

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/generate-panel-image", response_model=PanelImageResponse)
async def generate_panel_image(
    request: Request,
    payload: PanelRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
):
    """Generate a panel image for a webtoon."""
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    aspect_ratio: AspectRatioType = payload.aspectRatio or DEFAULT_ASPECT_RATIO
    resolution: ResolutionType = payload.resolution or DEFAULT_RESOLUTION
    analysis: AnalysisLevelType = payload.analysisLevel or "fast"
    cache_key = build_cache_key(payload, aspect_ratio, resolution, analysis)

    logger.info(
        "webtoon.generate-panel-image request panel=%s style=%s aspectRatio=%s resolution=%s analysisLevel=%s scene=%s dialogue=%s era=%s season=%s",
        payload.panelNumber,
        payload.style,
        aspect_ratio,
        resolution,
        analysis,
        payload.scene,
        payload.dialogue,
        payload.era,
        payload.season,
    )

    # Log additional metadata (excluding image data)
    logger.info(
        "webtoon.generate-panel-image metadata characters=%s characterDescriptions=%s characterImagesCount=%s referencesCount=%s revisionNote=%s",
        payload.characters,
        payload.characterDescriptions,
        len(payload.characterImages) if payload.characterImages else 0,
        len(payload.references) if payload.references else 0,
        payload.revisionNote,
    )

    if payload.characterSheetMetadata:
        sheet_metadata_summary = [
            {"name": entry.name, "hasMetadata": bool(entry.metadata)}
            for entry in payload.characterSheetMetadata
        ]
        logger.info(
            "webtoon.generate-panel-image characterSheetMetadata=%s",
            sheet_metadata_summary,
        )

    if payload.previousPanels:
        previous_panels_summary = [
            {
                "panel": p.get("panel"),
                "hasScene": bool(p.get("scene")),
                "hasDialogue": bool(p.get("dialogue")),
                "hasMetadata": bool(p.get("metadata")),
            }
            for p in payload.previousPanels
        ]
        logger.info(
            "webtoon.generate-panel-image previousPanels=%s",
            previous_panels_summary,
        )

    if payload.references:
        references_summary = [
            {"purpose": ref.purpose, "hasMimeType": bool(ref.mimeType)}
            for ref in payload.references
        ]
        logger.info(
            "webtoon.generate-panel-image references=%s",
            references_summary,
        )

    # Check for SSE streaming request
    accept_header = request.headers.get("accept", "")
    wants_stream = "text/event-stream" in accept_header

    cached = get_cached_panel_image(cache_key)
    if cached:
        if wants_stream:
            async def cached_stream() -> AsyncGenerator[str, None]:
                yield f"event: status\ndata: {json.dumps({'stage': 'cache', 'message': 'cache hit'}, ensure_ascii=False)}\n\n"
                result_data = cached.model_dump() if hasattr(cached, "model_dump") else cached.dict()
                yield f"event: result\ndata: {json.dumps(result_data, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                },
            )
        return cached

    if wants_stream:
        return StreamingResponse(
            generate_sse_stream(
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

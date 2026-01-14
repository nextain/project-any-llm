from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from any_llm import AnyLLM, acompletion
from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.chat import (
    _get_model_pricing,
    _get_provider_credentials,
    _log_usage,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .parser import parse_scene_response
from .prompt import (
    LANGUAGE_LABELS,
    build_prompt,
    build_scene_summary,
    build_world_setting_block,
    format_scene_elements,
    has_scene_elements,
    normalize_scene_elements,
    resolve_era_label,
    resolve_season_label,
)
from .schema import DEFAULT_MODEL, GeneratePanelSceneRequest, GeneratePanelSceneResponse

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/generate-panel-scene", response_model=GeneratePanelSceneResponse)
async def generate_panel_scene(
    request: GeneratePanelSceneRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> JSONResponse | GeneratePanelSceneResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    raw_elements = request.sceneElements.dict() if request.sceneElements else {}
    normalized_elements = normalize_scene_elements(raw_elements)
    fallback_scene = (request.baseScene or "").strip() or build_scene_summary(normalized_elements, "")
    if not fallback_scene:
        raise HTTPException(status_code=400, detail="Scene description or elements required")

    resolved_language = request.language or "ko"
    language_label = LANGUAGE_LABELS.get(resolved_language, LANGUAGE_LABELS["ko"])
    era_label = resolve_era_label(request.era)
    season_label = resolve_season_label(request.season)
    world_setting_block = build_world_setting_block(era_label, season_label)
    scene_elements_block = format_scene_elements(normalized_elements) if has_scene_elements(normalized_elements) else ""

    prompt = build_prompt(
        scene_elements_block=scene_elements_block,
        fallback_scene=fallback_scene,
        base_scene=(request.baseScene or "").strip() or fallback_scene,
        dialogue=request.dialogue,
        speaker=request.speaker,
        panel_number=request.panelNumber,
        topic=request.topic,
        genre=request.genre,
        style=request.style,
        language_label=language_label,
        world_setting_block=world_setting_block,
    )

    model_input = DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_credentials(config, provider)

    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": "You are a writer refining scene descriptions for a webtoon storyboard."},
            {"role": "user", "content": prompt},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        logger.info(
            "webtoon.panel-scene request panel=%s language=%s",
            request.panelNumber,
            resolved_language,
        )
        response = cast(Any, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/generate-panel-scene",
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
        scene_result = parse_scene_response(response)
        if not scene_result:
            logger.warning("Panel scene parser returned empty text, falling back to base scene.")
            return JSONResponse({"scene": fallback_scene})
        return GeneratePanelSceneResponse(scene=scene_result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Panel scene generation failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/generate-panel-scene",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Failed to generate panel scene") from exc

from __future__ import annotations

from typing import Any, cast

import re

from any_llm.types.completion import ChatCompletion
from fastapi import APIRouter, Depends, HTTPException, status
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

from .parser import extract_text_from_response, parse_script_response
from .prompt import (
    CARICATURE_GUIDE,
    GENRE_PUNCTUATION_GUIDES,
    GENRE_TONE_GUIDES,
    LANGUAGE_LABELS,
    build_system_prompt,
    build_user_prompt,
    build_world_setting_block,
    get_style_prompt,
    resolve_era_label,
    resolve_season_label,
)
from .schema import DEFAULT_MODEL, GenerateScriptRequest

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/script", response_model=None)
async def generate_script(
    request: GenerateScriptRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> Any:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    style_prompt = get_style_prompt(request.style)
    resolved_language = request.language or "ko"
    language_label = LANGUAGE_LABELS[resolved_language]
    tone_guide = GENRE_TONE_GUIDES.get(request.genre, "Use an appropriate tone for the chosen genre while following the rules above.")
    punctuation_guide = GENRE_PUNCTUATION_GUIDES.get(request.genre, "Use punctuation sparingly and only when it fits the tone.")
    era_label = resolve_era_label(request.era)
    season_label = resolve_season_label(request.season)
    world_setting_block = build_world_setting_block(era_label, season_label)
    user_prompt = build_user_prompt(
        topic=request.topic,
        style=request.style,
        genre=request.genre,
        normalized_elements=request.topicElements,
        language_label=language_label,
        tone_guide=tone_guide,
        punctuation_guide=punctuation_guide,
        caricature_mode=(request.characterGenerationMode or "ai") == "caricature",
        world_setting=world_setting_block,
        style_prompt=style_prompt,
    )

    model_input = DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_kwargs(config, provider)

    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": build_system_prompt(resolved_language, style_prompt)},
            {"role": "user", "content": user_prompt},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        logger.info(
            "webtoon.script request genre=%s style=%s language=%s era=%s season=%s mode=%s",
            request.genre,
            request.style,
            resolved_language,
            request.era,
            request.season,
            request.characterGenerationMode,
        )
        response = cast(ChatCompletion, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/script",
            user_id=user_id,
            response=cast(Any, response),
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
        text = extract_text_from_response(response)
        parsed = parse_script_response(text) if text else None
        if not parsed:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Invalid response structure from AI")
        return parsed
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Script generation failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/script",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Script generation failed") from exc

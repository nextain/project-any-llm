from __future__ import annotations

from typing import Any, cast

from any_llm.types.completion import ChatCompletion
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
    _get_provider_kwargs,
    _log_usage,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .parser import build_fallback_topics, extract_text_from_response, parse_topic_candidates
from .prompt import LANGUAGE_LABELS, build_prompt, build_system_prompt, resolve_era_label, resolve_genre_prompt, resolve_season_label
from .schema import DEFAULT_MODEL, GenerateTopicRequest, GenerateTopicResponse

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/topic", response_model=GenerateTopicResponse)
async def generate_topic(
    request: GenerateTopicRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> JSONResponse | GenerateTopicResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=request.user,
        missing_master_detail="When using master key, 'user' field is required in request body",
    )
    validate_user_credit(db, user_id)

    resolved_language = request.language or "ko"
    language_label = LANGUAGE_LABELS[resolved_language]
    genre_prompt = resolve_genre_prompt(request.genre)
    era_label = resolve_era_label(request.era)
    season_label = resolve_season_label(request.season)
    prompt = build_prompt(genre_prompt, language_label, era_label, season_label)

    model_input = request.model or DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_kwargs(config, provider)

    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": build_system_prompt(language_label)},
            {"role": "user", "content": prompt},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        logger.info(
            "webtoon.topic request model=%s genre=%s language=%s era=%s season=%s",
            model_input,
            request.genre,
            resolved_language,
            request.era,
            request.season,
        )
        response = cast(ChatCompletion, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/topic",
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
        text = extract_text_from_response(response)
        parsed = parse_topic_candidates(text) if text else None
        if not parsed:
            logger.error("Invalid response schema for topics")
            return build_fallback_topics(genre_prompt["title"], resolved_language)
        return parsed
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Topic generation failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/topic",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        return build_fallback_topics(genre_prompt["title"], resolved_language)

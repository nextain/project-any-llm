from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, status
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

from .parser import build_fallback_topic, extract_text_from_response, parse_topic_text
from .prompt import (
    LANGUAGE_LABELS,
    build_prompt,
    build_world_setting_block,
    normalize_scene_elements,
    resolve_era_label,
    resolve_season_label,
)
from .schema import DEFAULT_MODEL, GenerateTopicFromElementsRequest, GenerateTopicFromElementsResponse
from ..topic.prompt import SYSTEM_PROMPT_TEMPLATE

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


def build_system_prompt(language_label: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(language_label=language_label)


@router.post("/topic-from-elements", response_model=GenerateTopicFromElementsResponse)
async def generate_topic_from_elements(
    request: GenerateTopicFromElementsRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> JSONResponse | GenerateTopicFromElementsResponse:
    normalized_elements = normalize_scene_elements(request.sceneElements)
    fallback_topic = build_fallback_topic(normalized_elements)
    if not fallback_topic:
        return JSONResponse(content={"topic": ""}, status_code=status.HTTP_400_BAD_REQUEST)

    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    resolved_language = request.language or "ko"
    language_label = LANGUAGE_LABELS[resolved_language]
    era_label = resolve_era_label(request.era)
    season_label = resolve_season_label(request.season)
    world_setting_block = build_world_setting_block(era_label, season_label)
    prompt = build_prompt(
        topic=fallback_topic,
        normalized_elements=normalized_elements,
        genre=request.genre,
        language_label=language_label,
        world_setting_block=world_setting_block,
    )

    model_input = DEFAULT_MODEL
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
            "webtoon.topic-from-elements model=%s genre=%s language=%s era=%s season=%s",
            model_input,
            request.genre,
            resolved_language,
            request.era,
            request.season,
        )
        response = cast(Any, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/topic-from-elements",
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
        topic_text = parse_topic_text(text) if text else None
        if not topic_text:
            logger.error("Topic-from-elements response invalid, falling back")
            return GenerateTopicFromElementsResponse(topic=fallback_topic)
        return GenerateTopicFromElementsResponse(topic=topic_text)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Topic-from-elements generation failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/topic-from-elements",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        return GenerateTopicFromElementsResponse(topic=fallback_topic)

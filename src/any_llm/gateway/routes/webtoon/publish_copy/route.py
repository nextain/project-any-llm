from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
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

from .parser import clean_text, extract_text, parse_json
from .prompt import build_prompt
from .schema import DEFAULT_MODEL, GeneratePublishCopyRequest, GeneratePublishCopyResponse

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/generate-publish-copy", response_model=GeneratePublishCopyResponse)
async def generate_publish_copy(
    request: GeneratePublishCopyRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> GeneratePublishCopyResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    prompt = build_prompt(request.topic or "", request.genre or "", request.style or "", request.scriptSummary or "")
    model_input = DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_credentials(config, provider)

    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": "Write exhibition-ready copy."},
            {"role": "user", "content": prompt},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        logger.info("webtoon.generate-publish-copy request topic=%s", request.topic)
        response = cast(Any, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/generate-publish-copy",
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
        if not text:
            raise HTTPException(status_code=502, detail="Empty response from AI")
        cleaned = clean_text(text)
        parsed = parse_json(cleaned)
        if not parsed:
            raise HTTPException(status_code=502, detail="Invalid response structure")
        try:
            return GeneratePublishCopyResponse.model_validate(parsed)
        except Exception as exc:
            logger.warning("Publish copy validation failed: %s", exc)
            raise HTTPException(status_code=502, detail="Invalid response structure")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Publish copy generation failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/generate-publish-copy",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Publish copy generation failed") from exc

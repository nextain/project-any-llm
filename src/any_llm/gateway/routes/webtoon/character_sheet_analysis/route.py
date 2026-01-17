from __future__ import annotations

import base64
import re
from typing import Any, cast

from any_llm.types.completion import ChatCompletion
import httpx

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
    _get_provider_kwargs,
    _log_usage,
)
from any_llm.gateway.routes.utils import (
    charge_usage_cost,
    resolve_target_user,
    validate_user_credit,
)

from .parser import extract_text_from_response, parse_metadata
from .prompt import PROMPT
from .schema import CharacterSheetMetadata, CharacterSheetRequest, CharacterSheetResponse, DEFAULT_MODEL

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


def is_data_url(value: str) -> bool:
    return value.startswith("data:")


def parse_data_url(value: str) -> tuple[str, str]:
    match = re.match(r"^data:(.*?);base64,(.*)$", value)
    if not match:
        raise ValueError("Invalid data URL")
    mime_type, payload = match.groups()
    return mime_type, payload


async def fetch_image_as_base64(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/png")
        payload = base64.b64encode(response.content).decode("utf-8")
        return content_type, payload


@router.post("/character-sheet-analysis", response_model=CharacterSheetResponse)
async def analyze_character_sheet(
    request: CharacterSheetRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> CharacterSheetResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    if is_data_url(request.imageUrl):
        mime_type, payload = parse_data_url(request.imageUrl)
    else:
        mime_type, payload = await fetch_image_as_base64(request.imageUrl)

    model_input = DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_kwargs(config, provider)

    analysis_prompt = f"{PROMPT}\nImage MIME: {mime_type}\nImage Data: {payload}"
    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": "You analyze character sheets with precision."},
            {"role": "user", "content": analysis_prompt},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        response = cast(ChatCompletion, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/character-sheet-analysis",
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
        metadata = parse_metadata(text) if text else None
        if not metadata:
            metadata = CharacterSheetMetadata(
                summary=text or "",
                persona="",
                outfit=[],
                colors=[],
                accessories=[],
                hair="",
                face="",
                body="",
                props=[],
                shoes=[],
                notes=[],
            )
        return CharacterSheetResponse(metadata=metadata)
    except Exception as exc:
        logger.error("Character sheet analysis failed: %s", exc)
        raise HTTPException(status_code=502, detail="Character analysis failed") from exc

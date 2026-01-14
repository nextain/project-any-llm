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

from .parser import clean_text, extract_text_from_response, parse_json
from .prompt import build_prompt, build_previous_context, build_world_setting_block, resolve_era_label, resolve_season_label
from .schema import DEFAULT_MODEL, PanelHistoryEntry, RefinePanelScriptRequest, RefinePanelScriptResponse

router = APIRouter(prefix="/v1/webtoon", tags=["webtoon"])


@router.post("/refine-panel-script", response_model=RefinePanelScriptResponse)
async def refine_panel_script(
    request: RefinePanelScriptRequest,
    auth_result: tuple[APIKey | None, bool, str | None, SessionToken | None] = Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
) -> RefinePanelScriptResponse:
    api_key, _, _, _ = auth_result
    user_id = resolve_target_user(
        auth_result,
        explicit_user=None,
        missing_master_detail="When using master key, 'user' field is required",
    )
    validate_user_credit(db, user_id)

    era_label = resolve_era_label(request.era)
    season_label = resolve_season_label(request.season)
    world_setting_block = build_world_setting_block(era_label, season_label)

    previous_context_list = []
    for panel in request.previousPanels or []:
        index = panel.panel or panel.panelNumber or ""
        scene_text = panel.scene or ""
        dialogue_text = panel.dialogue or ""
        metadata_text = f" / Meta: {panel.metadata}" if panel.metadata else ""
        if scene_text or dialogue_text:
            previous_context_list.append(f"Panel {index}: {scene_text} / {dialogue_text}{metadata_text}")

    prompt_system, prompt_user = build_prompt(
        topic=request.topic or "",
        genre=request.genre or "",
        style=request.style or "",
        world_setting_block=world_setting_block,
        panel_number=request.panelNumber,
        speaker=request.speaker or "",
        scene=request.scene,
        dialogue=request.dialogue,
        improvement=request.improvement or "",
        revision_prompt=request.revisionPrompt or "",
        next_hint=request.nextHint or "",
        previous_context=build_previous_context(previous_context_list),
    )

    model_input = DEFAULT_MODEL
    provider, model = AnyLLM.split_model_provider(model_input)
    model_key, model_pricing = _get_model_pricing(db, provider, model)
    credentials = _get_provider_credentials(config, provider)

    completion_kwargs = {
        "model": model_input,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        "user": user_id,
        **credentials,
        "stream": False,
    }

    try:
        logger.info("webtoon.refine-panel-script request panel=%s", request.panelNumber)
        response = cast(Any, await acompletion(**completion_kwargs))
        usage_log_id = await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/refine-panel-script",
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
        if not text:
            logger.warning("Refine script response empty")
            raise HTTPException(status_code=502, detail="Empty response from AI")
        cleaned = clean_text(text)
        parsed = parse_json(cleaned)
        if not parsed:
            raise HTTPException(status_code=502, detail="Invalid response structure")
        try:
            return RefinePanelScriptResponse.model_validate(parsed)
        except Exception as exc:
            logger.warning("Refine script validation failed: %s", exc)
            raise HTTPException(status_code=502, detail="Invalid response structure")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Panel script refinement failed: %s", exc)
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/webtoon/refine-panel-script",
            user_id=user_id,
            model_key=model_key,
            model_pricing=model_pricing,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Panel script refinement failed") from exc

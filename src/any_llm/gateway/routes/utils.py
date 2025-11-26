from typing import Any, Tuple

from fastapi import HTTPException, status

from any_llm.gateway.db import APIKey, SessionToken


def resolve_target_user(
    auth_result: Tuple[APIKey | None, bool, str | None, SessionToken | None],
    explicit_user: str | None,
    *,
    missing_master_detail: str = "When using master key, user is required",
) -> str:
    """Resolve a target user_id from auth context and optional explicit value."""
    api_key, is_master, resolved_user_id, _ = auth_result

    if is_master:
        if not explicit_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=missing_master_detail,
            )
        return explicit_user

    target_user_id = resolved_user_id or explicit_user or (api_key.user_id if api_key else None)
    if not target_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not resolved")
    return target_user_id

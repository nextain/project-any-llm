import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth.models import hash_key
from any_llm.gateway.auth.tokens import verify_access_token
from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, get_db

_config: GatewayConfig | None = None


def set_config(config: GatewayConfig) -> None:
    """Set the global config instance."""
    global _config  # noqa: PLW0603
    _config = config


def get_config() -> GatewayConfig:
    """Get the global config instance."""
    if _config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return _config


def _extract_bearer_token(request: Request, config: GatewayConfig) -> str:
    """Extract and validate Bearer token from request header."""
    x_anyllm_key = request.headers.get(API_KEY_HEADER)

    if not x_anyllm_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER} header",
        )

    if not x_anyllm_key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {API_KEY_HEADER} header format. Expected 'Bearer <token>'",
        )

    return x_anyllm_key[7:]


def _verify_and_update_api_key(db: Session, token: str) -> APIKey:
    """Verify API key token and update last_used_at."""
    try:
        key_hash = hash_key(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key format: {e}",
        ) from e

    api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
        )

    if api_key.expires_at and api_key.expires_at < datetime.now(UTC).replace(tzinfo=None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    api_key.last_used_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()

    return api_key


def _is_valid_master_key(token: str, config: GatewayConfig) -> bool:
    """Check if token matches the master key."""
    return config.master_key is not None and secrets.compare_digest(token, config.master_key)


def _validate_session_token(db: Session, token: SessionToken) -> None:
    """Validate session token state."""
    now = datetime.now(UTC)
    if token.revoked_at:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session revoked",
        )
    if token.refresh_expires_at and token.refresh_expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
        )


async def verify_api_key(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> APIKey:
    """Verify API key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)
    return _verify_and_update_api_key(db, token)


async def verify_master_key(
    request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Verify master key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        config: Gateway configuration

    Raises:
        HTTPException: If master key is not configured or invalid

    """
    if not config.master_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Master key not configured. Set GATEWAY_MASTER_KEY environment variable.",
        )

    token = _extract_bearer_token(request, config)

    if not _is_valid_master_key(token, config):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid master key",
        )


async def verify_api_key_or_master_key(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> tuple[APIKey | None, bool]:
    """Verify either API key or master key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        Tuple of (APIKey object or None, is_master_key boolean)

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)

    if _is_valid_master_key(token, config):
        return None, True

    api_key = _verify_and_update_api_key(db, token)
    return api_key, False


async def verify_jwt_or_api_key_or_master(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> tuple[APIKey | None, bool, str | None, SessionToken | None]:
    """Verify JWT (access token), API key, or master key.

    Returns:
        (api_key_obj_or_none, is_master_key, user_id_or_none, session_token_or_none)
    """
    token = _extract_bearer_token(request, config)

    if _is_valid_master_key(token, config):
        return None, True, None, None

    # Try access token (JWT)
    try:
        payload = verify_access_token(token, config)
        jti = payload.get("jti")
        user_id = payload.get("sub")
        if not jti or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token payload",
            )

        session_token = db.query(SessionToken).filter(SessionToken.id == jti).first()
        if not session_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session not found",
            )
        _validate_session_token(db, session_token)

        session_token.last_used_at = datetime.now(UTC)
        db.commit()

        return None, False, str(user_id), session_token
    except HTTPException:
        raise
    except Exception:
        # Fall back to API key verification
        pass

    api_key = _verify_and_update_api_key(db, token)
    return api_key, False, str(api_key.user_id) if api_key.user_id else None, None

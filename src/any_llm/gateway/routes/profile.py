from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.db import APIKey, Budget, CaretUser, SessionToken, UsageLog, User, get_db
from any_llm.gateway.routes.utils import resolve_target_user

router = APIRouter(prefix="/v1/profile", tags=["profile"])


class BudgetInfo(BaseModel):
    """예산 정보."""

    budget_id: str | None
    max_budget: float | None
    budget_duration_sec: int | None
    spend: float
    budget_started_at: str | None
    next_budget_reset_at: str | None


class ProfileInfo(BaseModel):
    """사용자/소셜 통합 프로필."""

    user_id: str
    provider: str | None
    role: str | None
    name: str | None
    email: str | None
    avatar_url: str | None
    refresh_token: str | None
    blocked: bool
    spend: float
    metadata: dict
    caret_metadata: dict
    last_login_at: str | None


class UsageWindow(BaseModel):
    """기간별 사용량 합계."""

    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageLogItem(BaseModel):
    """최근 사용 로그 일부."""

    id: str
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None


class ProfileResponse(BaseModel):
    """프로필 응답."""

    profile: ProfileInfo
    usage: dict[str, UsageWindow]


class UsageBucket(BaseModel):
    """그룹별 집계."""

    period: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageBucketsResponse(BaseModel):
    """기간별 집계 응답."""

    group_by: str
    start: str
    end: str
    buckets: list[UsageBucket]
    total: UsageBucket


class KeySummary(BaseModel):
    """API 키 메타 요약(평문 키 미노출)."""

    id: str
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    metadata: dict


class LogItem(BaseModel):
    """사용 로그 아이템."""

    id: str
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None
    api_key_id: str | None
    user_id: str | None


class LogsResponse(BaseModel):
    """사용 로그 응답."""

    total: int
    items: list[LogItem]


def _now_naive() -> datetime:
    """UTC now without tzinfo (DB 저장 방식과 동일)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _ensure_naive(dt: datetime) -> datetime:
    """Ensure datetime is naive UTC."""
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _aggregate_usage(db: Session, user_id: str, since: datetime) -> UsageWindow:
    """기간별 사용량 합계."""
    requests_count, prompt_sum, completion_sum, total_sum, cost_sum = (
        db.query(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
        )
        .filter(UsageLog.user_id == user_id, UsageLog.timestamp >= since)
        .one()
    )

    return UsageWindow(
        requests=int(requests_count),
        prompt_tokens=int(prompt_sum or 0),
        completion_tokens=int(completion_sum or 0),
        total_tokens=int(total_sum or 0),
        cost=float(cost_sum or 0.0),
    )


def _recent_usage(db: Session, user_id: str, limit: int) -> list[UsageLogItem]:
    """최근 사용 로그."""
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user_id)
        .order_by(UsageLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        UsageLogItem(
            id=log.id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
        )
        for log in logs
    ]


@router.get("")
async def get_profile(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
) -> ProfileResponse:
    """프로필 + 예산 + 사용량 집계 반환."""
    api_key, is_master, resolved_user_id, session_token = auth_result
    target_user_id = resolve_target_user(
        (api_key, is_master, resolved_user_id),
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )
    user_obj = db.query(User).filter(User.user_id == target_user_id).first()
    if not user_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{target_user_id}' not found")

    caret = db.query(CaretUser).filter(CaretUser.user_id == target_user_id).first()

    now = _now_naive()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = start_of_day - timedelta(days=start_of_day.weekday())
    start_of_month = start_of_day.replace(day=1)
    usage_windows = {
        "today": _aggregate_range(db, target_user_id, start_of_day, now),
        "this_week": _aggregate_range(db, target_user_id, start_of_week, now),
        "this_month": _aggregate_range(db, target_user_id, start_of_month, now),
    }

    profile = ProfileInfo(
        user_id=user_obj.user_id,
        provider=caret.provider if caret else None,
        role=caret.role if caret else None,
        spend=float(user_obj.spend),
        name=caret.name if caret else user_obj.alias,
        email=caret.email if caret else None,
        avatar_url=caret.avatar_url if caret else None,
        refresh_token=session_token.refresh_token_plain if session_token else None,
        blocked=bool(user_obj.blocked),
        metadata=dict(user_obj.metadata_) if user_obj.metadata_ else {},
        caret_metadata=dict(caret.metadata_) if caret and caret.metadata_ else {},
        last_login_at=caret.last_login_at.isoformat() if caret and caret.last_login_at else None,
    )

    return ProfileResponse(
        profile=profile,
        usage=usage_windows,
    )


@router.get("/usage")
async def get_profile_usage(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
    start: datetime | None = Query(None, description="집계 시작 시각(ISO). 기본: now-30d"),
    end: datetime | None = Query(None, description="집계 종료 시각(ISO). 기본: now"),
    group_by: Literal["day", "week", "total"] = Query("day", description="집계 단위"),
) -> UsageBucketsResponse:
    """사용량 집계(기간별)."""
    target_user_id = resolve_target_user(
        auth_result[:3],
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

    now = _now_naive()
    start_dt = _ensure_naive(start) if start else now - timedelta(days=30)
    end_dt = _ensure_naive(end) if end else now
    if start_dt > end_dt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="start must be before end")

    total = _aggregate_range(db, target_user_id, start_dt, end_dt)

    if group_by == "total":
        buckets = [
            UsageBucket(
                period="total",
                requests=total.requests,
                prompt_tokens=total.prompt_tokens,
                completion_tokens=total.completion_tokens,
                total_tokens=total.total_tokens,
                cost=total.cost,
            )
        ]
    else:
        truncate_unit = "day" if group_by == "day" else "week"
        rows = (
            db.query(
                func.date_trunc(truncate_unit, UsageLog.timestamp).label("period"),
                func.count(UsageLog.id),
                func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
                func.coalesce(func.sum(UsageLog.completion_tokens), 0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                func.coalesce(func.sum(UsageLog.cost), 0.0),
            )
            .filter(UsageLog.user_id == target_user_id, UsageLog.timestamp >= start_dt, UsageLog.timestamp <= end_dt)
            .group_by("period")
            .order_by("period")
            .all()
        )

        buckets = [
            UsageBucket(
                period=row[0].isoformat(),
                requests=int(row[1]),
                prompt_tokens=int(row[2] or 0),
                completion_tokens=int(row[3] or 0),
                total_tokens=int(row[4] or 0),
                cost=float(row[5] or 0.0),
            )
            for row in rows
        ]

    return UsageBucketsResponse(
        group_by=group_by,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        buckets=buckets,
        total=UsageBucket(
            period="total",
            requests=total.requests,
            prompt_tokens=total.prompt_tokens,
            completion_tokens=total.completion_tokens,
            total_tokens=total.total_tokens,
            cost=total.cost,
        ),
    )


def _aggregate_range(db: Session, user_id: str, start: datetime, end: datetime) -> UsageWindow:
    """임의 기간 합계."""
    requests_count, prompt_sum, completion_sum, total_sum, cost_sum = (
        db.query(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
        )
        .filter(UsageLog.user_id == user_id, UsageLog.timestamp >= start, UsageLog.timestamp <= end)
        .one()
    )

    return UsageWindow(
        requests=int(requests_count),
        prompt_tokens=int(prompt_sum or 0),
        completion_tokens=int(completion_sum or 0),
        total_tokens=int(total_sum or 0),
        cost=float(cost_sum or 0.0),
    )


@router.get("/keys")
async def list_profile_keys(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
) -> list[KeySummary]:
    """사용자의 API 키 메타 조회(평문 키 미노출)."""
    target_user_id = resolve_target_user(
        auth_result[:3],
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

    keys = (
        db.query(APIKey)
        .filter(APIKey.user_id == target_user_id)
        .order_by(APIKey.created_at.asc())
        .all()
    )

    return [
        KeySummary(
            id=str(key.id),
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat() if key.created_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )
        for key in keys
    ]


@router.get("/logs")
async def list_profile_logs(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    start: datetime | None = Query(None, description="시작 시각(ISO)"),
    end: datetime | None = Query(None, description="종료 시각(ISO)"),
    status: str | None = Query(None, description="success 또는 error"),
    model: str | None = Query(None, description="모델 키 필터"),
    provider: str | None = Query(None, description="프로바이더 필터"),
    endpoint: str | None = Query(None, description="엔드포인트 필터"),
    min_prompt_tokens: int | None = Query(None),
    max_prompt_tokens: int | None = Query(None),
    min_completion_tokens: int | None = Query(None),
    max_completion_tokens: int | None = Query(None),
    min_total_tokens: int | None = Query(None),
    max_total_tokens: int | None = Query(None),
    min_cost: float | None = Query(None),
    max_cost: float | None = Query(None),
) -> LogsResponse:
    """사용자 로그 목록(필터/페이지네이션)."""
    target_user_id = resolve_target_user(
        auth_result,
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

    start_dt = _ensure_naive(start) if start else None
    end_dt = _ensure_naive(end) if end else None
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="start must be before end")

    query = db.query(UsageLog).filter(UsageLog.user_id == target_user_id)
    if start_dt:
        query = query.filter(UsageLog.timestamp >= start_dt)
    if end_dt:
        query = query.filter(UsageLog.timestamp <= end_dt)
    if status:
        query = query.filter(UsageLog.status == status)
    if model:
        query = query.filter(UsageLog.model == model)
    if provider:
        query = query.filter(UsageLog.provider == provider)
    if endpoint:
        query = query.filter(UsageLog.endpoint == endpoint)
    if min_prompt_tokens is not None:
        query = query.filter(UsageLog.prompt_tokens >= min_prompt_tokens)
    if max_prompt_tokens is not None:
        query = query.filter(UsageLog.prompt_tokens <= max_prompt_tokens)
    if min_completion_tokens is not None:
        query = query.filter(UsageLog.completion_tokens >= min_completion_tokens)
    if max_completion_tokens is not None:
        query = query.filter(UsageLog.completion_tokens <= max_completion_tokens)
    if min_total_tokens is not None:
        query = query.filter(UsageLog.total_tokens >= min_total_tokens)
    if max_total_tokens is not None:
        query = query.filter(UsageLog.total_tokens <= max_total_tokens)
    if min_cost is not None:
        query = query.filter(UsageLog.cost >= min_cost)
    if max_cost is not None:
        query = query.filter(UsageLog.cost <= max_cost)

    total = query.count()
    logs = (
        query.order_by(UsageLog.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        LogItem(
            id=log.id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            api_key_id=log.api_key_id,
            user_id=log.user_id,
        )
        for log in logs
    ]

    return LogsResponse(total=int(total), items=items)

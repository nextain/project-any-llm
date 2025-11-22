# /v1/users 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/users.py`
- 사용자 CRUD 및 사용 이력 조회를 제공. 모든 엔드포인트는 master key 인증(`verify_master_key`) 필요.
- 예산/사용량/키와 연계되는 핵심 사용자 관리 라우터.

## 데이터 모델
- `User` (`gateway/db/models.py`)
  - `user_id`(PK), `alias`, `spend`, `budget_id`, `budget_started_at`, `next_budget_reset_at`, `blocked`, `metadata`
  - 관계: `Budget`, `APIKey`, `UsageLog`, `BudgetResetLog`
- 예산 리셋 계산은 `gateway/budget.py`의 `calculate_next_reset`, `reset_user_budget`에서 수행.

## 엔드포인트

### POST `/v1/users`
- Request: `CreateUserRequest`
  - `user_id`(필수), `alias`, `budget_id`, `blocked`, `metadata`
- Response: `UserResponse`
- 동작:
  - 기존 `user_id`가 있으면 409.
  - `budget_id`가 있으면 Budget 존재 여부 검증, 시작 시점/다음 리셋 시간 설정.

### GET `/v1/users`
- Query: `skip`(기본 0), `limit`(기본 100)
- Response: `list[UserResponse]`

### GET `/v1/users/{user_id}`
- Response: `UserResponse`
- 없으면 404.

### PATCH `/v1/users/{user_id}`
- Request: `UpdateUserRequest`
  - `alias`, `budget_id`, `blocked`, `metadata` (모두 Optional)
- Response: `UserResponse`
- 동작:
  - 존재하지 않으면 404.
  - `budget_id` 변경 시 Budget 존재 여부 확인 후 `budget_started_at`/`next_budget_reset_at` 갱신(주기가 없으면 `next_budget_reset_at=None`).

### DELETE `/v1/users/{user_id}`
- Response: 204 No Content
- 없으면 404.

### GET `/v1/users/{user_id}/usage`
- Query: `skip`(기본 0), `limit`(기본 100)
- Response: `list[UsageLogResponse]`
  - 필드: `id, user_id, api_key_id, timestamp, model, provider, endpoint, prompt_tokens, completion_tokens, total_tokens, cost, status, error_message`
- 동작: 사용자 없으면 404. 최신순(desc) 페이지네이션.

## 연동/주의사항
- `spend`는 `/v1/chat/completions` 사용량 로깅에서 비용 계산 시 가산된다.
- `blocked=True`인 경우 `validate_user_budget`에서 요청을 거부한다.
- `budget_id`가 설정되면 `validate_user_budget`가 한도 초과/리셋을 처리한다.
- master key 헤더: `X-AnyLLM-Key: Bearer <master>`.

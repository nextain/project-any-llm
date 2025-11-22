# /v1/budgets 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/budgets.py`
- 예산 템플릿 CRUD를 제공하는 FastAPI 라우터.
- 모든 엔드포인트가 master key 인증(`verify_master_key`)을 요구한다.

## 데이터 모델
- SQLAlchemy `Budget` (`gateway/db/models.py`):
  - `budget_id`(UUID 문자열, PK)
  - `max_budget`(float | None): 최대 예산. None이면 한도 없음.
  - `budget_duration_sec`(int | None): 리셋 주기(초). None이면 자동 리셋 없음.
  - 타임스탬프: `created_at`, `updated_at`.

## 엔드포인트

### POST `/v1/budgets`
- Request: `CreateBudgetRequest`
  - `max_budget`: float | None
  - `budget_duration_sec`: int | None (예: 86400=일간, 604800=주간)
- Response: `BudgetResponse`
  - `budget_id`, `max_budget`, `budget_duration_sec`, `created_at`, `updated_at`
- 동작: 새 예산 생성 후 DB 커밋.

### GET `/v1/budgets`
- Query: `skip`(기본 0), `limit`(기본 100)
- Response: `list[BudgetResponse]`
- 동작: 페이지네이션 조회.

### GET `/v1/budgets/{budget_id}`
- Path: `budget_id`
- Response: `BudgetResponse`
- 동작: 존재하지 않으면 404.

### PATCH `/v1/budgets/{budget_id}`
- Request: `UpdateBudgetRequest`
  - `max_budget`: float | None
  - `budget_duration_sec`: int | None
- Response: `BudgetResponse`
- 동작: 필드가 제공된 것만 업데이트. 없으면 404.

### DELETE `/v1/budgets/{budget_id}`
- Response: 204 No Content
- 동작: 없으면 404, 있으면 삭제 후 커밋.

## 연동 포인트
- `Budget`는 사용자 예산 관리 로직(`gateway/budget.py`)에서 사용되며, `User.budget_id`와 연결되어 사용 금액 리셋·차단 판단에 활용된다.
- 실제 호출 시 master key(`X-AnyLLM-Key: Bearer <master>`)가 필요하다.

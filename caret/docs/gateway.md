# any-llm-gateway

- 코어 SDK 위에 키 관리, 예산, 가격, 사용량 로깅을 추가하는 FastAPI 프록시(`src/any_llm/gateway`).
- 설정(`GatewayConfig`, 위치: `src/any_llm/gateway/config.py`):
  - `database_url`, `auto_migrate`, `host`, `port`
  - `master_key` (관리자 엔드포인트에 필수)
  - `providers` 맵: 프로바이더 자격 증명(채팅 라우트에서 요청 생성 시 주입)
  - `pricing` 맵: `model_key -> 입력/출력 100만 토큰당 가격`, 시작 시 `pricing_init`이 DB에 반영
- CLI(`pyproject.toml`에 정의된 `any-llm-gateway`):
  - `serve`: FastAPI/uvicorn 실행. host/port/database_url/master_key/auto_migrate/workers/log level을 옵션으로 덮어쓴다.
  - `init_db`: 마이그레이션을 head까지 실행.
  - `migrate`: Alembic 업그레이드(타깃 리비전 선택 가능).

## 인증과 키

- 관리자 엔드포인트는 `X-AnyLLM-Key: Bearer <key>` 형태의 `master_key`가 필요.
- API 키는 `gw-<token>` 형태로 생성(`gateway/auth/models.py`), 저장 전 해시하며 `gateway/auth/dependencies.py`의 `verify_api_key`/`verify_api_key_or_master_key`로 검증.
- master key로 API 키(`/v1/keys`), 사용자(`/v1/users`), 예산(`/v1/budgets`), 가격(`/v1/pricing`)에 대한 생성/조회/수정/삭제가 가능.

## 예산과 사용량

- 예산 로직(`gateway/budget.py`): 사용자별 사용 금액 추적, `budget_duration_sec` 경과 시 자동 리셋, `max_budget` 초과 시 차단.
- 비용 계산: 채팅 라우트에서 `ModelPricing` 레코드를 기반으로 프롬프트/완료 토큰 × 단가를 산출해 `UsageLog`에 기록.
- 사용량 로깅은 스트리밍/비스트리밍 모두 수행하며, 스트리밍은 청크에 포함된 usage를 합산한다.

## API 개요

- `/v1/chat/completions` (`gateway/routes/chat.py`):
  - OpenAI 호환 페이로드.
  - `AnyLLM.split_model_provider`로 provider/model 파싱, 설정의 자격 증명 주입, SSE(`text/event-stream`) 스트리밍 지원.
  - 사용자 연동 강제(master key는 명시적 user 필요; API 키는 연결된/지정된 user 사용) 및 예산 검증 수행.
- `/v1/keys`: master-key 보호 CRUD, 사용자에 키 연결 또는 가상 사용자 자동 생성.
- `/v1/users`: master-key 보호 CRUD + `/v1/users/{user_id}/usage`로 사용 이력 조회.
- `/v1/budgets`: master-key 보호 예산 템플릿 CRUD.
- `/v1/pricing`: 모델 가격 설정/조회/삭제. DB에 없을 경우 시작 시 설정에서 부트스트랩.
- `/health`, `/health/liveness`, `/health/readiness`: 기본 상태/DB 연결성 체크.

## 데이터 모델(SQLAlchemy, `gateway/db/models.py`)

- `APIKey`, `User`, `Budget`, `UsageLog`, `ModelPricing`, `BudgetResetLog`로 구성되며 사용/리셋 이력을 관계로 관리.
- DB 세션 헬퍼와 마이그레이션 로직은 `gateway/db/session.py`, `gateway/alembic`에 위치.

## 프로바이더 헬퍼

- `VertexAuth`(`gateway/auth/vertex_auth.py`): 설정에 Vertex AI 자격 증명이 있을 때 `GOOGLE_APPLICATION_CREDENTIALS`/`GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION`을 세팅.
- `GatewayProvider`(`src/any_llm/providers/gateway/gateway.py`): `X-AnyLLM-Key` 헤더로 게이트웨이에 요청하는 SDK 측 클라이언트.

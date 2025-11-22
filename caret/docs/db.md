# DB 구조/연동/생성 흐름

## 테이블 정의 (`src/any_llm/gateway/db/models.py`)
- `APIKey`: 키 해시/상태/만료/메타데이터, `user_id` FK, `usage_logs` 관계.
- `Budget`: 예산 한도/주기, `users`, `reset_logs` 관계.
- `User`: `spend`, `budget_id`, `blocked`, `metadata`, `api_keys`, `usage_logs`, `reset_logs`.
- `ModelPricing`: `model_key`(provider:model)별 입력/출력 단가.
- `UsageLog`: 요청별 모델/프로바이더/엔드포인트, 토큰 수/비용, 상태/에러, `api_key`·`user` 관계.
- `BudgetResetLog`: 예산 리셋 이력(이전 사용량, 리셋 시각, 다음 리셋 시각).

## DB 초기화/마이그레이션
- `init_db(database_url, auto_migrate=True)` (`gateway/db/session.py`):
  - SQLAlchemy `create_engine` + `sessionmaker` 생성.
  - `auto_migrate=True`면 Alembic `upgrade head` 실행.
  - Alembic 설정 위치: `src/any_llm/gateway/alembic` (env.py, script.py.mako, alembic.ini 포함).
- 서버 기동 경로:
  - `any_llm.gateway.server.create_app` → `init_db(config.database_url, auto_migrate=config.auto_migrate)`.
  - `GatewayConfig.auto_migrate` 기본값 True, 비활성화하면 마이그레이션 건너뜀.
- CLI (`any-llm-gateway`):
  - `init_db`: `init_db()` 호출로 head까지 마이그레이션.
  - `migrate --revision <rev>`: Alembic upgrade 실행(기본 head).
  - `serve`: 앱 기동 시 `init_db` 자동 수행(옵션으로 auto_migrate 설정 가능).

## 세션 주입과 API 연동
- `get_db()` (`gateway/db/session.py`): 생성된 `_SessionLocal`로 세션을 열어 FastAPI Depends 제공.
- 모든 라우트에서 DB 접근 시 `db: Session = Depends(get_db)`로 주입:
  - 인증: `verify_api_key`, `verify_api_key_or_master_key`, `verify_master_key`가 세션을 사용해 키 검증/마지막 사용 시각 업데이트.
  - 키 관리(`/v1/keys`): APIKey 생성/조회/수정/삭제.
  - 사용자 관리(`/v1/users`): User CRUD, 예산 필드 업데이트.
  - 예산(`/v1/budgets`): Budget CRUD.
  - 가격(`/v1/pricing`): ModelPricing CRUD.
  - 채팅(`/v1/chat/completions`): UsageLog 기록 + ModelPricing 비용 계산 + User.spend 누적.
- 예산 로직(`gateway/budget.py`):
  - `validate_user_budget`가 User·Budget 조회, 한도 초과 차단, 주기 도래 시 `reset_user_budget` 실행(UsageLog와 별도).

## 연결 문자열/환경 변수
- `GatewayConfig.database_url` 기본값: `postgresql://postgres:postgres@localhost:5432/any_llm_gateway`.
- CLI `--database-url` 또는 환경변수 `DATABASE_URL`/`GATEWAY_DATABASE_URL`로 재정의 가능(Alembic 실행 시 env에 주입).
- master key/프로바이더 키와는 별개로 DB 계정 권한이 필요하므로 연결 문자열을 올바르게 설정해야 한다.

## Alembic 마이그레이션 버전 메모
- `versions/28d153c22616_initial_migration_with_all_models.py`: 초기 스키마 생성(APIKey, User, Budget, UsageLog, ModelPricing 등 핵심 테이블).
- `versions/1e382aa3a9e7_add_per_user_budget_resets.py`: 사용자별 예산 리셋 이력 관리를 위한 `budget_reset_logs` 추가 및 관련 FK/인덱스.
- `versions/e7c85cc73bfa_convert_budget_duration_to_seconds.py`: 예산 주기를 초 단위(`budget_duration_sec`)로 통일하는 변경.

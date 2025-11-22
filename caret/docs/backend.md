# 백엔드 구성 가이드 (any-llm-sdk 기반)

## 개요
- `demos/chat`, `demos/finder`는 프론트/백엔드 분리를 염두에 둔 예시 폴더이며, 실제 서버 코드는 `src` 아래 모듈을 조합해 작성할 수 있다.
- 백엔드 선택지:
  1) **직접 SDK 사용**: `any_llm.api` 또는 `AnyLLM` 클래스로 원하는 프레임워크(FastAPI 등)에 바로 통합.
  2) **게이트웨이 사용**: `src/any_llm/gateway`의 FastAPI 앱/라우터를 그대로 띄우거나 필요한 부분만 가져와 재사용.

## 구성 옵션 요약
- **직접 SDK** (`src/any_llm/api.py`):
  - 함수형 API: `completion/acompletion`, `responses/aresponses`, `embedding/aembedding`, `list_models/alist_models`.
  - 모델 문자열을 `provider:model`로 전달하거나 `provider` 인자를 별도로 전달.
  - 툴 호출 시 `prepare_tools`로 Python 함수를 OpenAI 툴 스키마로 변환 가능.
  - 비동기 API를 권장하며, 동기 필요 시 `run_async_in_sync`가 래핑.
- **게이트웨이 재사용** (`src/any_llm/gateway`):
  - 라우터: `/v1/chat/completions`, `/v1/keys`, `/v1/users`, `/v1/budgets`, `/v1/pricing`, `/health` 등.
  - DB 세션 주입, 예산 검증, 사용량/비용 로깅을 포함.
  - CLI `any-llm-gateway serve`로 전체 서버를 바로 실행 가능.
  - 필요 라우트만 가져와 의존성(`get_db`, 인증 deps 등)과 함께 조합할 수도 있다.

## 참고: 라우터 구성
- `src/any_llm/gateway/routes/chat.py`: OpenAI 호환 chat completions(SSE 스트리밍 지원), 예산/사용량 로깅.
- `src/any_llm/gateway/routes/keys.py`: API 키 CRUD (master key 필요).
- `src/any_llm/gateway/routes/users.py`: 사용자 CRUD + 사용 이력 조회.
- `src/any_llm/gateway/routes/budgets.py`: 예산 템플릿 CRUD.
- `src/any_llm/gateway/routes/pricing.py`: 모델별 단가 설정/조회.
- `src/any_llm/gateway/routes/health.py`: 헬스/레디니스 체크.

## 샘플: 최소 FastAPI 백엔드 (직접 SDK 통합)
- 비동기 `acompletion`을 사용해 단일 엔드포인트를 제공하는 예시.
- 요구 사항: `fastapi`, `uvicorn`, 선택한 프로바이더 extras(`pip install 'any-llm-sdk[openai]'` 등).

```python
# 샘플 파일: demos/chat/app.py (이미 추가됨)
# 실행: uvicorn demos.chat.app:app --reload --host 0.0.0.0 --port 8000
```

### 스트리밍을 추가하고 싶다면
- `result`가 `AsyncIterator[ChatCompletionChunk]`일 때 FastAPI의 `StreamingResponse`에 래핑하고, 청크를 `yield chunk.model_dump_json()` 형태로 보내면 된다.
- SSE를 원하면 `"data: ...\n\n"` 형태로 감싸고 `media_type="text/event-stream"`을 사용한다(`gateway/routes/chat.py` 참고).

### 툴 호출을 붙이고 싶다면
- Python 함수를 전달하거나, 이미 OpenAI 툴 스펙(dict)을 전달하면 된다.
- 예:
```python
from any_llm.tools import prepare_tools

def add(a: int, b: int) -> int:
    """두 수를 더한다."""
    return a + b

tools = prepare_tools([add])  # OpenAI tool schema 생성
result = await acompletion(model="openai:gpt-4o-mini", provider=None, messages=[...], tools=tools)
```

## 백엔드 설계 선택 가이드
- **간단한 단일 서비스**: 위의 미니 FastAPI 예제처럼 `any_llm.api` 직접 호출.
- **예산/키/사용량 관리 필요**: 게이트웨이 전체 또는 일부 라우터를 가져다 쓰기.
- **플랫폼 키로 멀티 프로바이더**: `PlatformProvider`를 사용해 `ANY_LLM_KEY`만 노출하고 실제 프로바이더 키를 숨긴다.
- **서드파티(OpenAI 호환) 엔드포인트 프록시**: `GatewayProvider` + 게이트웨이 라우터를 활용하면 OpenAI 호환 인터페이스로 통합 가능.

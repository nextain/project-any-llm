# any-llm-sdk 코어

- 진입점: `src/any_llm/__init__.py`에서 `AnyLLM`, `LLMProvider`, 동기/비동기 헬퍼(`completion/acompletion`, `responses/aresponses`, `embedding/aembedding`, `list_models/alist_models`, 배치 관련 함수)를 재노출한다.
- 사용 방식 두 가지:
  - `src/any_llm/api.py`의 무상태 헬퍼: `provider` 인자 또는 `provider:model` 문자열(레거시 `provider/model`도 경고와 함께 허용)을 받는다.
  - 재사용 클라이언트 클래스 `AnyLLM` (`src/any_llm/any_llm.py`): `AnyLLM.create(...)`로 생성하며, 프로바이더별 비동기 클라이언트를 재사용한다.
- 모든 동기 함수는 `run_async_in_sync`(`src/any_llm/utils/aio.py`)로 비동기 구현을 감싼다. 실행 중인 이벤트 루프(예: 노트북)에서는 스레드로 오프로드한다.

## 요청 처리 흐름

- 상위 API는 입력을 Pydantic 파라미터 모델(`CompletionParams`, `ResponsesParams`, 위치: `src/any_llm/types`)로 정규화한다.
- 프로바이더는 기능 플래그(`SUPPORTS_COMPLETION`, `SUPPORTS_RESPONSES` 등)를 선언하고, 비동기 훅(`_acompletion`, `_aresponses`, `_aembedding`, `_alist_models`, 배치 메서드)을 구현해야 한다. 동기 래퍼는 베이스 클래스에 있으며 이 훅들로만 전달한다.
- 툴 호출: `src/any_llm/tools.py`의 `prepare_tools`/`callable_to_tool`이 Python 함수 → OpenAI 툴 스키마로 변환한다. 빌트인 툴(예: Gemini의 `types.Tool`, `BUILT_IN_TOOLS`)도 그대로 통과시킨다.
- 모델 라우팅: `AnyLLM.split_model_provider`가 `provider:model`을 파싱하고 `LLMProvider` enum(`src/any_llm/constants.py`)으로 검증한다. 미지원 프로바이더는 `UnsupportedProviderError` 발생.
- API 키 처리: 각 프로바이더가 `ENV_API_KEY_NAME`을 정의하며, 키가 없으면 `MissingApiKeyError`. `ANY_LLM_KEY`를 넘기면 플랫폼 프로바이더가 any-llm 플랫폼에서 실제 키를 받아서 래핑한다.

## 응답/타입 모델

- `src/any_llm/types/completion.py`는 OpenAI 타입을 확장:
  - `ChatCompletionMessage.reasoning` 필드로 프로바이더 reasoning 내용을 분리(태그 추출 유틸은 `src/any_llm/utils/reasoning.py`).
  - 스트리밍 청크(`ChatCompletionChunk`, `ChoiceDelta`)도 reasoning 필드를 유지.
- Responses API는 OpenAI Responses 타입을 그대로 따라 `Response` 또는 `ResponseStreamEvent` 이터레이터를 반환(`src/any_llm/types/responses.py`).
- 임베딩/모델/배치 타입은 OpenAI 타입에 대한 얇은 별칭(`types/model.py`, `types/batch.py`).

## 배치 API (실험적)

- 클래스/모듈 레벨 둘 다 배치 생성·조회·취소 헬퍼가 있고 `experimental` 데코레이터(`src/any_llm/utils/decorators.py`)가 붙는다. `SUPPORTS_BATCH=True`인 프로바이더(OpenAI, gateway, platform 등)가 `_acreate_batch` 계열을 구현한다.

## 로깅과 에러

- `src/any_llm/logging.py`: Rich 핸들러로 `any_llm` 로거를 설정, `setup_logger`로 레벨/포맷 조정, 상위 전파 기본 차단.
- 주요 예외: `src/any_llm/exceptions.py`의 `MissingApiKeyError`, `UnsupportedProviderError`, `UnsupportedParameterError`.

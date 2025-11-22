# Internals and utilities

## 상수와 enum

- `LLMProvider`(`src/any_llm/constants.py`): 지원 프로바이더 문자열 enum. 검증과 메타데이터에 사용.
- `REASONING_FIELD_NAMES`: reasoning 유틸에서 소비하는 표준화된 태그 이름 목록.

## 툴 헬퍼

- `callable_to_tool` / `_python_type_to_json_schema` / `prepare_tools` (`src/any_llm/tools.py`):
  - 타입 힌트와 docstring이 있는 Python 함수를 OpenAI 툴 스키마로 변환.
  - TypedDict/dataclass/Pydantic 타입을 파라미터 스키마로 허용하며 Literal/Enum/sequence/dict/tuple 형태를 지원.
  - `prepare_tools`는 callable, 기존 툴 dict, 선택적 빌트인 툴을 혼합하여 반환.

## Async/sync 브리지

- `src/any_llm/utils/aio.py`의 `run_async_in_sync`, `async_iter_to_sync_iter`는 노트북/실행 중 루프에서도 스레드 오프로드와 pending task 정리를 통해 안전하게 동기 실행을 제공.

## Reasoning 정규화

- `src/any_llm/utils/reasoning.py`:
  - 프로바이더 전용 필드나 XML 태그(`<think>`, `<reasoning_content>` 등)에서 reasoning을 추출해 메시지 콘텐츠와 분리.
  - `process_streaming_reasoning_chunks`는 스트리밍 청크 사이에서 태그 단절을 버퍼링해 깨끗한 content + reasoning을 제공.

## 데코레이터와 경고

- `experimental`(`src/any_llm/utils/decorators.py`)은 sync/async 함수를 감싸 `FutureWarning`을 내보냄. 배치 API에 사용(`BATCH_API_EXPERIMENTAL_MESSAGE`).

## 로깅

- SDK 로거: `src/any_llm/logging.py`의 `any_llm`(Rich 핸들러, 레벨/포맷/전파 설정 가능).
- 게이트웨이 로거: `src/any_llm/gateway/log_config.py`의 `gateway`가 동일 패턴을 따름.

## 예외 타입

- `src/any_llm/exceptions.py`에 정의: `MissingApiKeyError`(키 없음), `UnsupportedProviderError`(잘못된 프로바이더), `UnsupportedParameterError`(프로바이더가 지원하지 않는 파라미터).

## 플랫폼 연동 헬퍼

- `src/any_llm/providers/platform/utils.py`: `ANY_LLM_KEY`를 파싱해 any-llm 플랫폼과 공개/비공개 키 챌린지를 수행, 프로바이더 키를 복호화하며 NaCl 기반 sealed-box 암호화를 사용해 `httpx`로 사용 이벤트를 전송.

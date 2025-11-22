# /v1/chat/completions 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/chat.py`
- OpenAI 호환 Chat Completions 엔드포인트를 FastAPI로 구현.
- 인증, 예산 검증, 프로바이더 자격 증명 주입, 사용량/비용 로깅, 스트리밍 SSE까지 처리.

## 요청 스키마 (`ChatCompletionRequest`)
- 필수: `model`(string), `messages`(list[dict])
- 선택: `user`, `temperature`, `max_tokens`, `top_p`, `stream`(bool, 기본 False), `tools`, `tool_choice`, `response_format`

## 인증 및 사용자 결정
- 인증은 `verify_api_key_or_master_key`로 처리:
  - 마스터 키라면 `user` 필드가 반드시 있어야 함. 없으면 400.
  - API 키라면 `user`를 명시하지 않은 경우 키에 연결된 사용자 ID를 사용. 연결이 없으면 500.
- 이후 `validate_user_budget`로 사용자 존재/차단 여부/예산 한도 검사 및 필요 시 예산 리셋.

## 프로바이더 선택과 자격 증명
- `AnyLLM.split_model_provider(request.model)`로 `(provider, model)` 분리.
- `config.providers`에서 해당 provider 설정을 찾아 `completion_kwargs`에 주입.
- Vertex AI인 경우 `setup_vertex_environment`가 GCP 자격 증명을 환경변수로 설정.

## 호출 흐름
- `completion_kwargs = request.model_dump()` 후 provider cred를 병합.
- `stream=False`:
  - `response = await acompletion(**completion_kwargs)`
  - `_log_usage`가 응답의 usage를 저장/비용 계산 후 DB 커밋.
- `stream=True`:
  - `stream = await acompletion(**completion_kwargs)` → AsyncIterator[ChatCompletionChunk]
  - 청크 순회하며:
    - usage 집계: prompt_tokens는 첫 비-0, completion_tokens/total_tokens는 최대값 유지.
    - SSE로 `data: <chunk_json>\n\n` 전송, 종료 시 `data: [DONE]\n\n`.
  - 스트림 종료 후 usage가 있으면 `_log_usage`로 기록. usage 없으면 경고만.

## 사용량/비용 로깅 (`_log_usage`)
- 성공/실패 모두 UsageLog를 남김(status: success/error).
- usage 소스:
  - 비스트리밍: 응답의 `response.usage`.
  - 스트리밍: 집계된 usage_override.
- 비용 계산: `model_key = f"{provider}:{model}"`, `ModelPricing`에 있으면 (prompt/1M * input_price + completion/1M * output_price)로 비용 산출.
  - user_id가 있으면 `User.spend`에 비용을 가산.
  - 가격 정보가 없으면 경고만 출력.

## 에러 처리
- 스트리밍/비스트리밍 모두 예외 시 `_log_usage`를 먼저 호출해 에러 상태로 기록.
- 이후 500 HTTPException(`"Error calling provider: <msg>"`)을 발생시킨다.

## 주의사항 / 팁
- 마스터 키로 호출 시 `user` 필드를 반드시 넣어야 한다.
- 모델 문자열은 `provider:model` 형식을 권장(미지정 시 provider를 추출할 수 없어 credentials 주입이 실패).
- 스트리밍은 SSE(`text/event-stream`) 형식이며, 프론트엔드에서 `data: [DONE]` 토큰으로 종료를 감지해야 한다.
- 가격/예산 기능을 쓰려면 DB에 `ModelPricing`, `Budget`이 필요하며, `pricing_init`가 설정을 DB에 반영하는 초기화가 선행돼야 한다.

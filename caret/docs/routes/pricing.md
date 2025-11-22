# /v1/pricing 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/pricing.py`
- 모델별 토큰 단가(입력/출력 100만 토큰 기준)를 설정/조회/삭제하는 라우터.
- master key 인증(`verify_master_key`)이 필요하다.

## 데이터 모델
- `ModelPricing` (`gateway/db/models.py`)
  - `model_key`: `provider:model` 문자열(PK)
  - `input_price_per_million`: float
  - `output_price_per_million`: float
  - 타임스탬프: `created_at`, `updated_at`
- 사용처: `/v1/chat/completions`에서 비용 계산 시 prompt/completion 토큰에 곱해 비용을 산출.

## 엔드포인트

### POST `/v1/pricing`
- Request: `SetPricingRequest`
  - `model_key` (예: `openai:gpt-4o-mini`)
  - `input_price_per_million`
  - `output_price_per_million`
- Response: `PricingResponse`
- 동작: 기존 레코드가 있으면 업데이트, 없으면 새로 생성 후 커밋/refresh.

### GET `/v1/pricing`
- Query: `skip`(기본 0), `limit`(기본 100)
- Response: `list[PricingResponse]`

### GET `/v1/pricing/{model_key}`
- Path: `model_key`
- Response: `PricingResponse`
- 없으면 404.

### DELETE `/v1/pricing/{model_key}`
- Response: 204 No Content
- 없으면 404, 있으면 삭제 후 커밋.

## 연동/주의사항
- 비용 계산은 채팅 라우트에서 `ModelPricing`을 조회해 `(prompt_tokens/1e6)*input_price + (completion_tokens/1e6)*output_price`로 적용한다.
- 가격이 없으면 비용 계산은 생략되며 경고만 로그에 남는다.
- master key 헤더: `X-AnyLLM-Key: Bearer <master>`.

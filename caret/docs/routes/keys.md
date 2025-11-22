# /v1/keys 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/keys.py`
- API 키 관리 CRUD를 제공. 모든 엔드포인트는 master key 인증(`verify_master_key`) 필요.
- 키 생성 시 필요하면 사용자도 함께 생성(또는 기존 사용자에 연결).

## 모델
- DB 모델: `APIKey` (`gateway/db/models.py`)
  - `id`, `key_hash`, `key_name`, `user_id`, `expires_at`, `is_active`, `metadata`, 타임스탬프, 관계(`User`, `UsageLog`)
- 키 포맷: `gw-<random>` (`generate_api_key`), 저장 시 SHA-256 해시(`hash_key`).

## 엔드포인트

### POST `/v1/keys`
- Request: `CreateKeyRequest`
  - `key_name`: Optional
  - `user_id`: Optional(지정 시 해당 사용자에 연결, 없으면 새 사용자 생성)
  - `expires_at`: Optional
  - `metadata`: dict, 기본 {}
- Response: `CreateKeyResponse`
  - 반환에 **평문 키** 포함(`key` 필드). 이후에는 해시만 저장되므로 최초 응답을 반드시 보관해야 함.
- 동작:
  - `user_id`가 없으면 `apikey-<uuid>`로 가상 사용자 생성.
  - 키 해시 저장 후 커밋.

### GET `/v1/keys`
- Query: `skip`(기본 0), `limit`(기본 100)
- Response: `list[KeyInfo]` (키 해시는 반환하지 않음)

### GET `/v1/keys/{key_id}`
- Path: `key_id`
- Response: `KeyInfo`
- 존재하지 않으면 404.

### PATCH `/v1/keys/{key_id}`
- Request: `UpdateKeyRequest`
  - `key_name`, `is_active`, `expires_at`, `metadata` (모두 Optional)
- Response: `KeyInfo`
- 존재하지 않으면 404.

### DELETE `/v1/keys/{key_id}`
- Response: 204 No Content
- 존재하지 않으면 404.

## 검증/보안 포인트
- 모든 엔드포인트가 master key를 요구: 헤더 `X-AnyLLM-Key: Bearer <master>`.
- 키 값은 생성 시점에만 평문으로 노출되므로 클라이언트가 안전하게 저장해야 한다.
- `UpdateKeyRequest`의 `is_active`와 `expires_at`으로 키를 비활성/만료 관리 가능.

## 관련 로직
- API 키 검증 및 사용 시점 업데이트는 `gateway/auth/dependencies.py`의 `verify_api_key`, `verify_api_key_or_master_key`가 담당(다른 라우트에서 사용).
- 키에 연결된 사용자/가상 사용자에 대해 사용량 로깅(`UsageLog`)과 비용 계산이 연계된다.

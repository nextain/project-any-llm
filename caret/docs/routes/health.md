# /health 라우트 정리

## 개요
- 위치: `src/any_llm/gateway/routes/health.py`
- 게이트웨이의 상태/준비 여부를 확인하기 위한 경량 헬스 체크 엔드포인트 모음.
- 인증 필요 없음.

## 엔드포인트

### GET `/health`
- 응답: `{"status": "healthy"}`
- 단순 가용성 체크용.

### GET `/health/liveness`
- 응답: 문자열 `"I'm alive!"`
- 프로세스 생존 여부만 확인(쿠버네티스 liveness probe용).

### GET `/health/readiness`
- 동작:
  - DB 연결 확인: `get_db()`로 세션 생성 후 `SELECT 1` 실행.
  - 성공 시 DB 상태 `connected`로 설정.
  - 예외 발생 시 503 반환, 상세 오류 포함.
- 응답(성공 시):
  ```json
  {
    "status": "healthy",
    "database": "connected",
    "version": "<__version__>"
  }
  ```
- 목적: 서비스가 요청을 처리할 준비가 되었는지 확인(쿠버네티스 readiness probe용).

## 예외/에러 처리
- DB 연결/쿼리 실패 시 HTTP 503 + 상세 에러 정보를 담아 반환.
- `__version__`은 `any_llm.gateway.__version__`에서 로드.

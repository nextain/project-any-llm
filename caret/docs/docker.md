# Docker / Compose 가이드

## Dockerfile 요약 (`docker/Dockerfile`)
- 베이스: `python:3.13-slim`.
- 빌드 컨텍스트에 `pyproject.toml`, `src/` 복사 후 extras `.[all,gateway]` 설치.
- `SETUPTOOLS_SCM_PRETEND_VERSION` ARG/ENV로 버전 주입(기본 `0.0.0+docker`).
- 비루트 사용자 `gateway`(uid 1000)로 실행, 작업 디렉토리 `/app`.
- 포트: 8000 노출. 헬스체크: `http://localhost:8000/health`.
- 기본 CMD: `any-llm-gateway serve` (env `GATEWAY_HOST=0.0.0.0`, `GATEWAY_PORT=8000`).

## Compose 구성 (`docker/docker-compose.yml`)
- `gateway` 서비스:
  - 빌드: 루트 컨텍스트, `docker/Dockerfile`, `VERSION` ARG(기본 `0.0.0+local`).
  - 포트 매핑: `8000:8000` (필요 시 `config.yml`과 함께 변경).
  - 볼륨: `./config.yml` → `/app/config.yml`, `./service_account.json` → `/app/service_account.json`.
  - 커맨드: `any-llm-gateway serve --config /app/config.yml`.
  - Postgres 서비스가 healthy일 때만 시작, `restart: unless-stopped`.
- `postgres` 서비스:
  - 이미지 `postgres:16-alpine`.
  - env: `POSTGRES_USER=gateway`, `POSTGRES_PASSWORD=gateway`, `POSTGRES_DB=gateway`.
  - 볼륨: `postgres_data`에 데이터 영구화.
  - 헬스체크: `pg_isready -U gateway`.

## 설정 예시 (`docker/config.example.yml`)
- DB: `database_url: postgresql://gateway:gateway@postgres:5432/gateway`.
- 서버: `host`, `port` (Compose 포트 매핑과 일치 필요).
- `master_key`: 키 관리 등 관리자 엔드포인트 보호용.
- `providers`: 프로바이더별 자격증명. Vertex AI는 서비스 계정 JSON 경로, `project`, `location` 필요.
- `pricing`: `"provider:model"` → `input_price_per_million`, `output_price_per_million` (USD). 초기값만 세팅; DB에 값이 있으면 우선.

## 사용 방법 요약
1. `docker/config.example.yml`를 복사해 `docker/config.yml` 작성, `master_key`와 필요한 프로바이더 키/경로를 채운다. Vertex AI를 쓰면 `service_account.json`을 `docker/`에 두고 Compose 볼륨과 맞춘다.
2. 필요한 경우 포트를 `config.yml`과 `docker-compose.yml` 모두에서 수정한다.
3. 빌드/실행: `docker compose -f docker/docker-compose.yml up --build`.
4. 헬스체크는 컨테이너 내부 `/health`를 주기적으로 확인한다(외부에서 확인하려면 호스트 포트로 호출).

## 이미지를 바로 쓰고 싶다면
- `docker-compose.yml`의 `build` 섹션을 주석 처리하고 `image: ghcr.io/mozilla-ai/any-llm/gateway:latest`를 활성화하면 레지스트리 이미지를 바로 사용 가능하다.

## 실행 요약 (게이트웨이)
- 콘솔 스크립트: `pyproject.toml`의 `[project.scripts]`에서 `any-llm-gateway = "any_llm.gateway.cli:main"`으로 등록되어 있으며, `cli.py`의 `serve` 커맨드를 실행한다.
- 로컬 실행(패키지 설치 후):
  - 기본 포트 8000, 기본 host 0.0.0.0. 필요 시 `GATEWAY_HOST`, `GATEWAY_PORT`, `GATEWAY_DATABASE_URL`로 오버라이드.
  - 설정 파일 사용 시: `any-llm-gateway serve --config /path/to/config.yml`
  - 설정 파일 없이: `any-llm-gateway serve`
- Docker 단독:
  - `docker run -p 8000:8000 -e GATEWAY_HOST=0.0.0.0 -e GATEWAY_PORT=8000 <image> any-llm-gateway serve`
- Compose:
  - `docker compose -f docker/docker-compose.yml up --build`
  - `docker/config.yml`의 `port`와 Compose 포트 매핑(`8000:8000`)이 일치해야 한다.

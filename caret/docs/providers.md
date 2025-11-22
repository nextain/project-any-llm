# Providers

- 모든 프로바이더는 `AnyLLM`(`src/any_llm/any_llm.py`)을 상속해 기능 플래그를 선언하고 `_init_client` 및 변환 헬퍼를 구현한다.
- 네이밍 규칙: 디렉터리와 클래스는 `<provider>/<ProviderName>Provider`; `AnyLLM.create`가 `any_llm.providers.<provider>`를 동적 import 한다.
- 공통 베이스:
  - `BaseOpenAIProvider`(`src/any_llm/providers/openai/base.py`): OpenAI 호환 JSON 형태를 사용하며, 기본적으로 completion/stream/embedding/list_models를 지원. 일부 하위 클래스가 Responses·batch·이미지·PDF 지원을 켜서 확장한다.
  - `GoogleProvider`(`src/any_llm/providers/gemini/base.py`): Gemini/Vertex AI 어댑터, 툴 변환과 reasoning budget, 빌트인 툴(`BUILT_IN_TOOLS=[types.Tool]`)을 통과시킨다.
  - `PlatformProvider`(`src/any_llm/providers/platform/platform.py`): `ANY_LLM_KEY`로 any-llm 플랫폼에서 실제 프로바이더 키를 받아 내부 프로바이더를 감싼 뒤 사용 이벤트를 전송한다.
  - `GatewayProvider`(`src/any_llm/providers/gateway/gateway.py`): any-llm-gateway와 통신하는 OpenAI 호환 클라이언트(`X-AnyLLM-Key` 헤더 사용).
- 선택적 의존성: 각 프로바이더는 `MISSING_PACKAGES_ERROR`로 미설치 시 에러를 던지며, `pip install any-llm-sdk[<provider>]`로 extras를 설치한다(`pyproject.toml` 참고).

## 기능 매트릭스

- Y/N 값은 프로바이더 플래그(베이스 클래스 기본값 포함) 기준. reasoning은 명시적 reasoning 필드 처리 여부, image/pdf는 해당 페이로드로 completion을 지원하는지 여부를 나타낸다.

| provider | env var | completion | stream | embedding | responses | list models | batch | reasoning | image | pdf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | ANTHROPIC_API_KEY | Y | Y | N | N | Y | N | Y | Y | N |
| azure | AZURE_API_KEY | Y | Y | Y | N | N | N | N | N | N |
| azureopenai | AZURE_OPENAI_API_KEY | Y | Y | Y | Y | Y | N | N | Y | Y |
| bedrock | AWS_BEARER_TOKEN_BEDROCK | Y | Y | Y | N | Y | N | Y | N | N |
| cerebras | CEREBRAS_API_KEY | Y | Y | N | N | Y | N | Y | N | N |
| cohere | COHERE_API_KEY | Y | Y | N | N | Y | N | Y | N | N |
| databricks | DATABRICKS_TOKEN | Y | Y | Y | N | N | N | Y | N | N |
| deepseek | DEEPSEEK_API_KEY | Y | Y | N | N | Y | N | Y | N | N |
| fireworks | FIREWORKS_API_KEY | Y | Y | N | Y | Y | N | Y | Y | N |
| gateway | GATEWAY_API_KEY | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| gemini | GEMINI_API_KEY / GOOGLE_API_KEY | Y | Y | Y | N | Y | N | Y | N | N |
| groq | GROQ_API_KEY | Y | Y | N | Y | Y | N | Y | N | N |
| huggingface | HF_TOKEN | Y | Y | N | N | Y | N | Y | N | N |
| inception | INCEPTION_API_KEY | Y | Y | N | N | Y | N | N | N | N |
| llama | LLAMA_API_KEY | Y | Y | N | N | Y | N | N | N | N |
| llamacpp | (none) | Y | Y | Y | N | Y | N | Y | Y | N |
| llamafile | (none) | Y | N | N | N | Y | N | Y | N | N |
| lmstudio | LM_STUDIO_API_KEY | Y | Y | Y | N | Y | N | Y | Y | Y |
| minimax | MINIMAX_API_KEY | Y | Y | N | N | N | N | Y | N | N |
| mistral | MISTRAL_API_KEY | Y | Y | Y | N | Y | N | Y | N | N |
| moonshot | MOONSHOT_API_KEY | Y | Y | N | N | Y | N | Y | N | N |
| nebius | NEBIUS_API_KEY | Y | Y | Y | N | Y | N | Y | Y | N |
| ollama | (none) | Y | Y | Y | N | Y | N | Y | Y | Y |
| openai | OPENAI_API_KEY | Y | Y | Y | Y | Y | Y | N | Y | Y |
| openrouter | OPENROUTER_API_KEY | Y | Y | N | N | Y | N | Y | Y | Y |
| perplexity | PERPLEXITY_API_KEY | Y | Y | N | N | N | N | N | Y | N |
| platform | ANY_LLM_KEY | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| portkey | PORTKEY_API_KEY | Y | Y | N | N | Y | N | Y | Y | Y |
| sagemaker | AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY | Y | Y | Y | N | N | N | N | Y | Y |
| sambanova | SAMBANOVA_API_KEY | Y | Y | Y | N | Y | N | Y | Y | N |
| together | TOGETHER_API_KEY | Y | Y | N | N | N | N | Y | Y | N |
| vertexai | GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION | Y | Y | Y | N | Y | N | Y | N | N |
| voyage | VOYAGE_API_KEY | N (임베딩 전용) | N | Y | N | N | N | N | N | N |
| watsonx | WATSONX_API_KEY | Y | Y | N | N | Y | N | N | Y | N |
| xai | XAI_API_KEY | Y | Y | N | N | Y | N | Y | N | N |
| zai | ZAI_API_KEY | Y | Y | N | N | Y | N | Y | N | N |

## 참고 동작

- reasoning 처리: 일부 프로바이더는 전용 필드나 XML 태그로 reasoning을 보낸다. OpenAI 호환 어댑터와 커스텀 프로바이더(예: Hugging Face)에서 정규화 유틸을 적용한다.
- 스트리밍: `stream=True`일 때 비동기 이터레이터를 반환하며, 퍼블릭 API가 동기 제너레이터로 변환한다. 청크에 usage가 포함되면 스트리밍 사용량을 집계한다.
- 응답 파싱: OpenAI 호환 어댑터가 프로바이더별 응답을 OpenAI 스키마(`_convert_completion_response`/`_convert_completion_chunk_response`)로 변환하고 object 이름, timestamp 등 불일치를 보정한다.

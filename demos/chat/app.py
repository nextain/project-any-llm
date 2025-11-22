"""Minimal FastAPI chat backend using any-llm-sdk directly."""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from any_llm.api import acompletion

app = FastAPI(title="any-llm minimal chat backend")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    provider: str | None = None
    stream: bool | None = False


async def _sse_stream(chunks: AsyncIterator) -> AsyncIterator[str]:
    """Wrap streaming chunks as SSE lines."""
    async for chunk in chunks:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    """Proxy chat completions to the selected provider."""
    try:
        result = await acompletion(
            model=req.model,
            provider=req.provider,  # None이면 model 문자열에서 provider:model을 파싱
            messages=req.messages,
            stream=req.stream,
        )
    except Exception as exc:  # noqa: BLE001 broad for demo
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 스트리밍 요청이면 SSE로 변환
    if req.stream:
        if not hasattr(result, "__aiter__"):
            raise HTTPException(status_code=400, detail="Provider did not return a stream.")
        return StreamingResponse(_sse_stream(result), media_type="text/event-stream")

    return result


# 실행 예시:
# uvicorn demos.chat.app:app --reload --host 0.0.0.0 --port 8000

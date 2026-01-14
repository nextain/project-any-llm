from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PlatformCopy(BaseModel):
    caption: str
    hashtags: List[str]


class GenerateSnsCopyRequest(BaseModel):
    topic: str | None = None
    genre: str | None = None
    scriptSummary: str | None = None


class GenerateSnsCopyResponse(BaseModel):
    facebook: PlatformCopy
    instagram: PlatformCopy
    threads: PlatformCopy


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

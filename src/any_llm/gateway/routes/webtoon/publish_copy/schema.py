from __future__ import annotations

from typing import List

from pydantic import BaseModel


class GeneratePublishCopyRequest(BaseModel):
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    scriptSummary: str | None = None


class PublishCopyEntry(BaseModel):
    title: str
    description: str


class GeneratePublishCopyResponse(BaseModel):
    title: str
    description: str


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

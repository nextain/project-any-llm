from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PanelReviewEntry(BaseModel):
    panel: int | None = None
    panelNumber: int | None = None
    scene: str | None = None
    speaker: str | None = None
    dialogue: str | None = None
    metadata: str | None = None


class ReviewWebtoonRequest(BaseModel):
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    scriptSummary: str | None = None
    panels: List[PanelReviewEntry] | None = None


class ReviewNextIdea(BaseModel):
    title: str
    topic: str
    genre: str
    style: str
    hook: str


class ReviewWebtoonResponse(BaseModel):
    headline: str
    summary: str
    strengths: List[str]
    improvements: List[str]
    encouragement: str
    nextIdeas: List[ReviewNextIdea]


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

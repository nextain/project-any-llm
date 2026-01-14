from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PanelHistoryEntry(BaseModel):
    panel: int | None = None
    panelNumber: int | None = None
    scene: str | None = None
    metadata: str | None = None


class PanelReviewRequest(BaseModel):
    panelNumber: int | None = None
    scene: str | None = None
    dialogue: str | None = None
    speaker: str | None = None
    metadata: str | None = None
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    era: str | None = None
    season: str | None = None
    previousPanels: list[PanelHistoryEntry] | None = None
    language: Literal["ko", "zh", "ja"] | None = None


class PanelReviewResponse(BaseModel):
    praise: str
    highlight: str
    improvement: str
    nextHint: str
    revisionPrompt: str
    badge: str


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

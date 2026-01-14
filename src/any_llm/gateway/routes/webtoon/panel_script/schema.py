from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PanelHistoryEntry(BaseModel):
    panel: int | None = None
    panelNumber: int | None = None
    scene: str | None = None
    dialogue: str | None = None
    metadata: str | None = None


class RefinePanelScriptRequest(BaseModel):
    panelNumber: int | None = None
    scene: str
    dialogue: str
    speaker: str | None = None
    improvement: str | None = None
    revisionPrompt: str | None = None
    nextHint: str | None = None
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    era: str | None = None
    season: str | None = None
    previousPanels: list[PanelHistoryEntry] | None = None
    language: Literal["ko", "zh", "ja"] | None = None


class RefinePanelScriptResponse(BaseModel):
    scene: str
    dialogue: str


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Language = Literal["ko", "zh", "ja"]

SCENE_ELEMENT_KEYS = ["subject", "action", "setting", "composition", "lighting", "style"]


class SceneElements(BaseModel):
    subject: str | None = None
    action: str | None = None
    setting: str | None = None
    composition: str | None = None
    lighting: str | None = None
    style: str | None = None


class GeneratePanelSceneRequest(BaseModel):
    panelNumber: int | None = None
    baseScene: str | None = None
    sceneElements: SceneElements | None = None
    dialogue: str | None = None
    speaker: str | None = None
    topic: str | None = None
    genre: str | None = None
    style: str | None = None
    era: str | None = None
    season: str | None = None
    language: Language | None = None


class GeneratePanelSceneResponse(BaseModel):
    scene: str


DEFAULT_MODEL = "gemini:gemini-2.5-flash"

from __future__ import annotations

from typing import Literal, List

from pydantic import BaseModel


class SceneElements(BaseModel):
    subject: str | None = None
    action: str | None = None
    setting: str | None = None
    composition: str | None = None
    lighting: str | None = None
    style: str | None = None


class GenerateCaricatureSheetRequest(BaseModel):
    name: str
    description: str | None = None
    style: str
    styleDoc: str | None = None
    strength: Literal["low", "medium", "high"] | None = None
    referenceImage: str
    era: str | None = None
    season: str | None = None
    sceneElements: SceneElements | None = None


class CaricatureSheetEntry(BaseModel):
    imageUrl: str | None
    metadata: str


class GenerateCaricatureSheetResponse(BaseModel):
    sheet: CaricatureSheetEntry

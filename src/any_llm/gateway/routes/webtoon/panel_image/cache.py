"""Caching functions for panel image generation."""
from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from .constants import (
    CACHE_TTL_SECONDS,
    IMAGE_CACHE,
    MAX_CACHE_ENTRIES,
    PROMPT_VERSION,
)
from .utils import _normalize_scene_elements

if TYPE_CHECKING:
    from .schema import (
        AnalysisLevelType,
        AspectRatioType,
        PanelImageResponse,
        PanelRequest,
        ResolutionType,
    )


def build_cache_key(
    body: "PanelRequest",
    aspect_ratio: "AspectRatioType",
    resolution: "ResolutionType",
    analysis_level: "AnalysisLevelType",
) -> str:
    """Build a unique cache key for a panel generation request."""
    hash_builder = hashlib.sha256()
    scene_elements_value = _normalize_scene_elements(body.sceneElements) if body.sceneElements is not None else None
    core_payload = {
        "scene": body.scene,
        "dialogue": body.dialogue or "",
        "characters": body.characters,
        "style": body.style,
        "panelNumber": body.panelNumber,
        "era": body.era or None,
        "season": body.season or None,
        "sceneElements": scene_elements_value,
        "characterDescriptions": body.characterDescriptions or [],
        "styleDoc": body.styleDoc or "",
        "previousPanels": body.previousPanels or [],
        "characterSheetMetadata": [
            entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
            for entry in body.characterSheetMetadata or []
        ],
        "characterGenerationMode": body.characterGenerationMode or None,
        "characterCaricatureStrengths": body.characterCaricatureStrengths or [],
        "revisionNote": body.revisionNote or "",
        "aspectRatio": aspect_ratio,
        "resolution": resolution,
        "analysisLevel": analysis_level,
        "promptVersion": PROMPT_VERSION,
    }
    hash_builder.update(json.dumps(core_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for image_url in body.characterImages or []:
        hash_builder.update(b"|character-image|")
        hash_builder.update((image_url or "").encode("utf-8"))
    for ref in body.references or []:
        hash_builder.update(b"|reference|")
        hash_builder.update((ref.base64 or "").encode("utf-8"))
        hash_builder.update((ref.mimeType or "").encode("utf-8"))
        hash_builder.update((ref.purpose or "").encode("utf-8"))
    return hash_builder.hexdigest()


def get_cached_panel_image(key: str) -> "PanelImageResponse | None":
    """Get a cached panel image response by key."""
    entry = IMAGE_CACHE.get(key)
    if not entry:
        return None
    payload, expires = entry
    if expires < time.time():
        del IMAGE_CACHE[key]
        return None
    return payload


def set_cached_panel_image(key: str, payload: "PanelImageResponse") -> None:
    """Cache a panel image response."""
    IMAGE_CACHE[key] = (payload, time.time() + CACHE_TTL_SECONDS)
    if len(IMAGE_CACHE) > MAX_CACHE_ENTRIES:
        oldest_key = next(iter(IMAGE_CACHE))
        del IMAGE_CACHE[oldest_key]


def finalize_response(
    payload_text: str,
    inline_image_base64: str,
    mime_type: str,
    aspect_ratio: "AspectRatioType",
    resolution: "ResolutionType",
    panel_number: int,
) -> "PanelImageResponse":
    """Create a finalized panel image response."""
    from .schema import PanelImageResponse

    return PanelImageResponse(
        success=True,
        imageUrl=f"data:{mime_type};base64,{inline_image_base64}",
        imageBase64=inline_image_base64,
        mimeType=mime_type,
        metadata=payload_text,
        text=payload_text,
        aspectRatio=aspect_ratio,
        resolution=resolution,
        model="gemini-3-pro-image-preview",
        panelNumber=panel_number,
    )

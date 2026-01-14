from __future__ import annotations

from typing import Literal

from .schema import GenerateCaricatureSheetRequest

CARICATURE_STRENGTH_GUIDE: dict[Literal["low", "medium", "high"], str] = {
    "low": "Light caricature. Keep the likeness recognizable but enlarge the head slightly (about 10-15%) and simplify shapes.",
    "medium": "Clear caricature. Enlarge the head (about 15-25%), emphasize 2-3 distinctive facial features (eyes, nose, jawline), and reduce realistic textures.",
    "high": "Bold caricature. Make the exaggeration obvious: enlarge the head (about 25-35%), push facial proportions and expression, and simplify body details.",
}

STYLE_GUIDE = """- Professional 2D vector flat illustration with bold, clean outlines.
- Smooth gradients and soft cel-shading; crisp edges.
- Simplified shapes with minimal texture; avoid realism.
- Clear silhouette and friendly proportions; keep a modern, approachable vibe.
- Head and facial features should be noticeably larger than the torso; simplify limbs."""

WORLD_SETTING_GUIDANCE = """- If an era is provided, the outfit and accessories must clearly reflect it.
- Season should adjust layers, fabrics, and color palette without overriding the era.
- If the art style implies a different era, keep the rendering style but prioritize the chosen era.
- Keep facial features aligned with the reference image.
- Keep the background plain as required.
- Avoid anachronistic items that break the chosen era."""


def build_prompt(request: GenerateCaricatureSheetRequest, mime_type: str) -> str:
    strength = request.strength or "medium"
    guide = CARICATURE_STRENGTH_GUIDE.get(strength, CARICATURE_STRENGTH_GUIDE["medium"])
    lines = [
        "Professional 2D vector flat illustration, high-quality caricature.",
        "Style: Minimalist character design, bold clean outlines, smooth gradients, soft cel-shading.",
        "Emphasize facial features while keeping a friendly, charismatic expression.",
        "Background: Solid neutral color. No text, no logos, no extra objects.",
        "Technical: 8K resolution, crisp edges, Dribbble quality.",
        "Reference image MIME type: " + mime_type,
    ]

    description = request.description or "the person in the reference image"
    subject = f"Name: {request.name}. Description: {description}."
    instructions = [
        STYLE_GUIDE,
        f"Strength guidance: {guide}",
        subject,
        WORLD_SETTING_GUIDANCE,
    ]

    if request.styleDoc:
        instructions.append(f"Style notes: {request.styleDoc}")

    if request.sceneElements:
        elements = []
        for key, label in (
            ("subject", "Subject"),
            ("action", "Action"),
            ("setting", "Setting"),
            ("composition", "Composition"),
            ("lighting", "Lighting"),
            ("style", "Scene Style"),
        ):
            value = getattr(request.sceneElements, key)
            if value:
                elements.append(f"- {label}: {value}")
        if elements:
            instructions.append("Scene context:\n" + "\n".join(elements))

    return "\n\n".join(lines + instructions)

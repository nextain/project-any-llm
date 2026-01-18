"""Constants and configuration for panel image generation."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import AspectRatioType, ResolutionType

# Cache settings
IMAGE_CACHE: dict[str, tuple] = {}
MAX_CACHE_ENTRIES = 50
CACHE_TTL_SECONDS = 5 * 60

# Default values
DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_RESOLUTION: "ResolutionType" = "1K"
DEFAULT_ASPECT_RATIO: "AspectRatioType" = "1:1"
PROMPT_VERSION = "scene-dialogue-v1"

# Resolution mappings
RESOLUTION_LONG_EDGE: dict[str, int] = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}

# Style prompts for different art styles
STYLE_PROMPTS: dict[str, dict[str, str]] = {
    "webtoon": {
        "id": "webtoon",
        "name": "웹툰 스타일",
        "systemPrompt": "당신은 한국 웹툰 전문 이미지 생성 AI입니다. 네이버 웹툰, 카카오웹툰 스타일의 깔끔한 선화와 디지털 채색을 사용합니다.",
        "imagePrompt": "Korean webtoon style, clean line art, digital coloring, vibrant colors, modern illustration, professional manhwa art, detailed character design, smooth shading, clear outlines, anatomically correct hands and limbs",
        "negativePrompt": "messy lines, sketch style, rough draft, watercolor, oil painting, realistic photo, 3D render, blurry, low quality, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "manga": {
        "id": "manga",
        "name": "만화 스타일",
        "systemPrompt": "당신은 일본 만화 전문 이미지 생성 AI입니다. 세밀한 스크린톤, 다양한 선의 강약, 감정 표현이 풍부한 일본 망가 스타일을 사용합니다.",
        "imagePrompt": "Japanese manga style, detailed screen tones, expressive line work, dynamic composition, black and white ink art, shounen/shoujo manga aesthetic, detailed backgrounds, varied line weights, anatomically correct hands and limbs",
        "negativePrompt": "colored, full color, digital painting, western comic style, realistic, photo, 3D, blurry, low detail, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "cartoon": {
        "id": "cartoon",
        "name": "카툰 스타일",
        "systemPrompt": "당신은 귀여운 카툰 캐릭터 전문 이미지 생성 AI입니다. 단순하고 둥근 형태, 밝고 선명한 색상, 친근하고 귀여운 느낌의 캐릭터를 만듭니다.",
        "imagePrompt": "Cute cartoon style, simple rounded shapes, bright vibrant colors, friendly character design, kawaii aesthetic, chibi proportions, clean flat colors, playful illustration, anatomically correct hands and limbs",
        "negativePrompt": "realistic, detailed, complex, dark, gritty, serious, photo-realistic, 3D render, sketch, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "illustration": {
        "id": "illustration",
        "name": "일러스트",
        "systemPrompt": "당신은 감성적인 일러스트 전문 이미지 생성 AI입니다. 디테일한 표현, 부드러운 빛과 그림자, 감정이 느껴지는 색감을 사용합니다.",
        "imagePrompt": "Detailed digital illustration, soft lighting, emotional atmosphere, painterly style, artistic composition, rich textures, sophisticated color palette, professional book illustration quality, anatomically correct hands and limbs",
        "negativePrompt": "simple, cartoon, chibi, low detail, flat colors, sketch, rough, unfinished, photo, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "realistic": {
        "id": "realistic",
        "name": "실사 스타일",
        "systemPrompt": "당신은 실사 스타일 이미지 생성 AI입니다. 사진처럼 사실적인 질감, 자연스러운 조명, 현실감 있는 표현을 사용합니다.",
        "imagePrompt": "Photorealistic style, realistic textures, natural lighting, high detail photography, professional photo quality, cinematic composition, realistic shadows and highlights, anatomically correct hands and limbs",
        "negativePrompt": "cartoon, anime, illustration, drawn, painted, sketch, abstract, stylized, flat, low quality, blurry, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
    "3d": {
        "id": "3d",
        "name": "3D 렌더링",
        "systemPrompt": "당신은 3D 렌더링 전문 이미지 생성 AI입니다. Pixar, Disney 스타일의 입체적이고 부드러운 3D 캐릭터와 환경을 만듭니다.",
        "imagePrompt": "3D rendered style, Pixar quality, smooth 3D models, professional rendering, volumetric lighting, detailed textures, Disney/Pixar aesthetic, clean 3D animation style, anatomically correct hands and limbs",
        "negativePrompt": "2D, flat, hand-drawn, sketch, photo, realistic, anime style, low poly, draft quality, extra limbs, extra hands, extra arms, extra fingers, mutated hands, fused fingers, too many fingers, malformed limbs, anatomical errors, duplicate body parts",
    },
}

# Scene element keys and labels
SCENE_ELEMENT_KEYS = ["subject", "action", "setting", "composition", "lighting", "style"]
SCENE_ELEMENT_LABELS = {
    "subject": "Subject(주제)",
    "action": "Action(동작)",
    "setting": "Setting(환경)",
    "composition": "Composition(구성/카메라)",
    "lighting": "Lighting(조명)",
    "style": "Style(스타일)",
}

# Caricature style guide
CARICATURE_PANEL_STYLE_GUIDE = """
- Professional 2D vector-like caricature style.
- Bold, clean outlines with simplified shapes.
- Soft cel-shading and smooth gradients; avoid painterly textures.
- Make the head and facial features noticeably larger than the torso; simplify limbs.
- Exaggerate facial proportions while keeping a friendly, charismatic expression.
- Maintain a consistent caricature identity across panels.
""".strip()

# Era guardrails for historical consistency
ERA_GUARDRAILS: dict[str, dict[str, list[str]]] = {
    "joseon": {
        "must": [
            "traditional wooden or stone architecture",
            "historical cooking tools (iron pot, brazier, earthenware)",
        ],
        "avoid": ["electric appliances", "cars", "glass skyscrapers", "smartphones", "modern signage"],
    },
    "nineties": {
        "must": ["CRT TV or bulky monitor", "landline phone or pager", "retro kitchen appliances"],
        "avoid": ["smartphones", "tablets", "flat-screen TVs", "wireless earbuds", "ultra-minimal interiors"],
    },
    "seventies-eighties": {
        "must": ["simple vintage interior", "analog appliances", "enamel pots or coal briquette tools"],
        "avoid": ["smartphones", "laptops", "flat-screen TVs", "induction stoves", "modern smart devices"],
    },
    "future": {
        "must": ["futuristic devices or interfaces", "sleek high-tech materials"],
        "avoid": ["purely antique tools", "old CRT TVs", "hand-cranked devices"],
    },
}

# Anatomical constraints for image generation
ANATOMICAL_CONSTRAINTS = """
ANATOMICAL ACCURACY - ABSOLUTE REQUIREMENTS:
- Each human character must have exactly: 2 arms, 2 hands with 5 fingers each, 2 legs, 1 head.
- Never generate extra limbs, duplicate body parts, or fused appendages.
- Hands must be clearly separated and anatomically correct.
- If a character's hands are not visible in the scene, do not force them into frame.
- Avoid complex overlapping poses that may cause limb confusion.
""".strip()

ANATOMICAL_NEGATIVE_PROMPT = (
    "extra limbs, extra hands, extra arms, extra fingers, extra legs, "
    "mutated hands, fused fingers, too many fingers, malformed limbs, "
    "missing fingers, deformed hands, anatomical errors, body horror, "
    "duplicate body parts, conjoined limbs, twisted anatomy, "
    "unnatural pose, impossible anatomy, broken wrists"
)

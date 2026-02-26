"""TTS endpoint — proxies Google Cloud Text-to-Speech via Nextain credits.

Supports two auth modes:
1. Service account (Cloud Run default) — no API key needed
2. API key fallback — uses GOOGLE_TTS_API_KEY or GEMINI_API_KEY
"""

import base64
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, SessionToken, UsageLog, User, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.routes.utils import (
    charge_user_credits,
    get_user_credits_per_usd,
    resolve_target_user,
    validate_user_credit,
)

router = APIRouter(prefix="/v1/audio", tags=["audio"])

# Google Cloud TTS pricing (USD per 1M characters)
_PRICING = {
    "standard": 4.0,
    "wavenet": 16.0,
    "neural2": 16.0,
}

_AUDIO_ENCODING_MAP = {
    "MP3": "MP3",
    "OGG_OPUS": "OGG_OPUS",
    "LINEAR16": "LINEAR16",
    "MULAW": "MULAW",
    "ALAW": "ALAW",
}

# Lazy-loaded TTS client (service account auth on Cloud Run)
_tts_client = None


def _get_tts_client():
    """Get or create Google Cloud TTS client (uses ADC / service account)."""
    global _tts_client
    if _tts_client is None:
        from google.cloud import texttospeech
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=5000)
    voice: str = "ko-KR-Neural2-A"
    language_code: str | None = None
    audio_encoding: str = "MP3"
    speaking_rate: float = 1.0
    pitch: float = 0.0


class SpeechResponse(BaseModel):
    audio_content: str
    character_count: int
    cost_usd: float


def _derive_language_code(voice: str) -> str:
    """Derive language code from voice name (e.g. 'ko-KR-Neural2-A' -> 'ko-KR')."""
    parts = voice.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return "ko-KR"


def _voice_tier(voice: str) -> str:
    """Determine pricing tier from voice name."""
    lower = voice.lower()
    if "neural2" in lower:
        return "neural2"
    if "wavenet" in lower:
        return "wavenet"
    return "standard"


def _log_tts_usage(
    db: Session,
    api_key_obj: APIKey | None,
    user_id: str | None,
    character_count: int,
    voice: str,
    error: str | None = None,
) -> str | None:
    """Insert a UsageLog row for TTS usage."""
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_obj.id if api_key_obj else None,
        user_id=user_id,
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        model=f"tts-{_voice_tier(voice)}",
        provider="google_tts",
        endpoint="/v1/audio/speech",
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=character_count,
        completion_tokens=0,
        total_tokens=character_count,
        cached_tokens=0,
    )
    db.add(usage_log)
    try:
        db.commit()
        return usage_log.id
    except Exception as exc:
        logger.error("Failed to log TTS usage: %s", str(exc))
        db.rollback()
        return None


async def _call_gcp_tts(
    text: str,
    voice: str,
    language_code: str,
    audio_encoding: str,
    speaking_rate: float,
    pitch: float,
    db: Session,
    api_key_obj: APIKey | None,
    user_id: str | None,
) -> tuple[str, float, str | None]:
    """Call Google Cloud TTS via service account and handle billing. Returns (audio_b64, cost_usd, usage_id)."""
    import asyncio
    from google.cloud import texttospeech

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice,
    )
    gcp_encoding = getattr(
        texttospeech.AudioEncoding,
        _AUDIO_ENCODING_MAP.get(audio_encoding, "MP3"),
        texttospeech.AudioEncoding.MP3,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=gcp_encoding,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )

    try:
        client = _get_tts_client()
        response = await asyncio.to_thread(
            client.synthesize_speech,
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
    except Exception as exc:
        error_msg = str(exc)[:300]
        _log_tts_usage(db, api_key_obj, user_id, len(text), voice, error=error_msg)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Google TTS error: {error_msg}",
        )

    audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")

    char_count = len(text)
    tier = _voice_tier(voice)
    rate = _PRICING.get(tier, 16.0)
    cost_usd = (char_count / 1_000_000) * rate

    usage_id = _log_tts_usage(db, api_key_obj, user_id, char_count, voice)

    if cost_usd > 0:
        try:
            credits_per_usd = get_user_credits_per_usd(db, user_id)
            charge_user_credits(
                db, user_id=user_id, cost_usd=cost_usd,
                credits_per_usd=credits_per_usd,
                model_key=f"google_tts:tts-{tier}",
                usage_id=usage_id,
            )
        except Exception as e:
            logger.warning("TTS credit charge failed", extra={"user_id": user_id, "error": str(e)})

    if usage_id:
        try:
            log_row = db.query(UsageLog).filter(UsageLog.id == usage_id).first()
            if log_row:
                log_row.cost = cost_usd
                db.commit()
        except Exception:
            db.rollback()

    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user and cost_usd > 0:
            user.spend = (user.spend or 0) + cost_usd
            db.commit()
    except Exception:
        db.rollback()

    return audio_b64, cost_usd, usage_id


@router.post("/speech")
async def synthesize_speech(
    request: Request,
    auth_result=Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
):
    """Synthesize speech via Google Cloud TTS.

    Auto-detects request format:
    - OpenAI format (has "model" field): returns raw audio bytes
    - Nextain format (has "audio_encoding" field): returns JSON with base64
    """
    api_key_obj, is_master, _, _ = auth_result
    user_id = resolve_target_user(auth_result, None)
    validate_user_credit(db, user_id)

    body = await request.json()
    is_openai_format = "model" in body

    if is_openai_format:
        req = OpenAISpeechRequest(**body)
        gcp_voice = req.voice if "-" in req.voice else "ko-KR-Neural2-A"
        language_code = _derive_language_code(gcp_voice)
        audio_encoding = _FORMAT_MAP.get(req.response_format, "MP3")

        audio_b64, _, _ = await _call_gcp_tts(
            req.input, gcp_voice, language_code,
            audio_encoding, req.speed, 0.0,
            db, api_key_obj, user_id,
        )

        audio_bytes = base64.b64decode(audio_b64)
        content_type = _CONTENT_TYPE_MAP.get(req.response_format, "audio/mpeg")
        return Response(content=audio_bytes, media_type=content_type)

    req = SpeechRequest(**body)
    language_code = req.language_code or _derive_language_code(req.voice)
    audio_b64, cost_usd, _ = await _call_gcp_tts(
        req.input, req.voice, language_code,
        req.audio_encoding, req.speaking_rate, req.pitch,
        db, api_key_obj, user_id,
    )

    return SpeechResponse(
        audio_content=audio_b64,
        character_count=len(req.input),
        cost_usd=cost_usd,
    )


# --- OpenAI-compatible TTS endpoint ---
# Accepts OpenAI's request format and returns raw audio bytes,
# enabling OpenClaw and other OpenAI-compatible clients to use Nextain TTS.

# Map OpenAI response_format → Google Cloud TTS audioEncoding
_FORMAT_MAP = {
    "mp3": "MP3",
    "opus": "OGG_OPUS",
    "aac": "MP3",       # GCP doesn't support AAC; fall back to MP3
    "flac": "LINEAR16",  # closest GCP equivalent
    "wav": "LINEAR16",
    "pcm": "LINEAR16",
}

_CONTENT_TYPE_MAP = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/mpeg",
    "flac": "audio/x-wav",
    "wav": "audio/x-wav",
    "pcm": "audio/x-wav",
}


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible TTS request body."""

    model: str = "tts-1"
    input: str = Field(min_length=1, max_length=5000)
    voice: str = "ko-KR-Neural2-A"
    response_format: str = "mp3"
    speed: float = 1.0


@router.post("/speech/openai")
async def openai_compatible_speech(
    req: OpenAISpeechRequest,
    auth_result=Depends(verify_jwt_or_api_key_or_master),
    db: Session = Depends(get_db),
    config: GatewayConfig = Depends(get_config),
):
    """OpenAI-compatible TTS: accepts OpenAI format, returns raw audio bytes."""
    api_key_obj, is_master, _, _ = auth_result
    user_id = resolve_target_user(auth_result, None)
    validate_user_credit(db, user_id)

    gcp_voice = req.voice if "-" in req.voice else "ko-KR-Neural2-A"
    language_code = _derive_language_code(gcp_voice)
    audio_encoding = _FORMAT_MAP.get(req.response_format, "MP3")

    audio_b64, _, _ = await _call_gcp_tts(
        req.input, gcp_voice, language_code,
        audio_encoding, req.speed, 0.0,
        db, api_key_obj, user_id,
    )

    audio_bytes = base64.b64decode(audio_b64)
    content_type = _CONTENT_TYPE_MAP.get(req.response_format, "audio/mpeg")
    return Response(content=audio_bytes, media_type=content_type)

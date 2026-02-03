from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from app.auth import verify_api_key
from app.audio_utils import save_base64_audio
from app.detector import detect_voice
from app.config import SUPPORTED_LANGUAGES


app = FastAPI(
    title="AI vs Human Voice Detector",
    version="2.0",
    description="Detect whether a voice sample is AI-generated or Human using acoustic analysis"
)


class DetectRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        description="Base64-encoded MP3 audio (single voice sample)"
    )
    language: str = Field(
        ...,
        description="Language code: en, hi, ta, te, ml"
    )


class DetectResponse(BaseModel):
    classification: str = Field(
        ..., description="AI_GENERATED | HUMAN | UNCERTAIN"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score"
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of the decision"
    )


@app.post("/detect", response_model=DetectResponse)
def detect(
    request: DetectRequest,
    _=Depends(verify_api_key)
):
    # --- Validate language ---
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported languages: {SUPPORTED_LANGUAGES}"
        )

    try:
        # --- Decode audio ---
        audio_path = save_base64_audio(request.audio_base64)

        # --- Run detection (inference only) ---
        classification, confidence, explanation = detect_voice(
            audio_path, request.language
        )

    except ValueError as e:
        # Controlled validation / feature errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        # Prevent leaking internal errors to judges
        raise HTTPException(
            status_code=500,
            detail="Internal server error during audio analysis"
        )

    return DetectResponse(
        classification=classification,
        confidence=confidence,
        explanation=explanation
    )

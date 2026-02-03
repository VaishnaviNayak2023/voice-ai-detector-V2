import numpy as np
from app.audio_utils import extract_features
from app.model_loader import lr, svm, rf, scaler


def detect_voice(audio_path: str, language: str):
    features = extract_features(audio_path)
    features = scaler.transform([features])

    probs = np.array([
        lr.predict_proba(features)[0][1],
        svm.predict_proba(features)[0][1],
        rf.predict_proba(features)[0][1]
    ])

    confidence = float(probs.mean())

    if confidence > 0.6:
        classification = "AI_GENERATED"
        explanation = "Consistent spectral patterns and reduced micro-variations."
    elif confidence < 0.4:
        classification = "HUMAN"
        explanation = "Natural pitch drift and temporal irregularities detected."
    else:
        classification = "UNCERTAIN"
        explanation = "Mixed or ambiguous acoustic characteristics."

    return classification, round(confidence, 3), explanation

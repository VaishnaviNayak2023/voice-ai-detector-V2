import base64
import tempfile
import librosa
import numpy as np


MIN_DURATION_SEC = 0.5
MIN_RMS_ENERGY = 1e-4
SAMPLE_RATE = 16000
N_MFCC = 13


def save_base64_audio(base64_audio: str) -> str:
    """
    Decode Base64 MP3 and save to temp file
    """
    try:
        audio_bytes = base64.b64decode(base64_audio)
    except Exception:
        raise ValueError("Invalid Base64 audio")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def extract_features(path: str) -> np.ndarray:
    """
    Feature extraction used during inference
    MUST exactly match training features
    """
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    # --- Edge case: very short audio ---
    if len(y) < sr * MIN_DURATION_SEC:
        raise ValueError("Audio too short for reliable analysis")

    # --- Edge case: silence / no speech ---
    rms_energy = np.mean(librosa.feature.rms(y=y))
    if rms_energy < MIN_RMS_ENERGY:
        raise ValueError("Audio contains little or no speech")

    # --- Core acoustic features ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # --- Feature vector (ORDER MATTERS) ---
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(spectral_centroid),
        np.mean(spectral_rolloff),
        np.mean(zcr),
        np.mean(rms)
    ])

    return features.astype(np.float32)

import librosa
import numpy as np

def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=16000)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1),
        delta.mean(axis=1),
        [zcr.mean()],
        [rms.mean()],
        [spec_centroid.mean()]
    ])

    return features

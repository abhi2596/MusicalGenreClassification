import librosa
import numpy as np


# refer to paper for the features to be extracted
def extract_features(signal):

    result = []
    features = []

    spectral_centroid = librosa.feature.spectral_centroid(y=signal)
    spectral_rollof = librosa.feature.spectral_rolloff(y=signal)
    spectral_flux = librosa.onset.onset_strength(y=signal)
    zero_crossing = librosa.feature.zero_crossing_rate(y=signal)
    rms = librosa.feature.rms(y=signal)
    mfcc = librosa.feature.mfcc(y=signal)

    features.append(spectral_centroid)
    features.append(spectral_rollof)
    features.append(spectral_flux)
    features.append(zero_crossing)
    features.append(rms)

    for feature in features:
        result.append(np.mean(feature))
        result.append(np.std(feature))

    for i in range(0, 5):
        result.append(np.mean(mfcc[i, :]))
        result.append(np.std(mfcc[i, :]))

    return result

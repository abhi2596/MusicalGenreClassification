import librosa
import os
import numpy as np
from extract_features import extract_features


def create_dataset():
    genre_directory_path = os.path.join(os.getcwd(),"genres")
    labels = os.listdir(genre_directory_path)
    dataset = None
    target = []
    for label in labels:
        data_path = os.path.join(genre_directory_path, label)
        for file in os.listdir(data_path):
            audio_path = os.path.join(data_path, file)
            audio_sample, sr = librosa.load(audio_path)
            audio_sample = np.array(audio_sample)
            feature = extract_features(audio_sample)
            if dataset is None:
                dataset = np.array(feature)
            else:
                dataset = np.vstack((dataset, feature))
            target.append(label)
    return dataset, target

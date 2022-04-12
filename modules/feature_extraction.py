import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import librosa.display


# expecting dataframe with unlabeled samples dataset
# returns dataframe with added audio features
def extract_features(df_raw_dataset):
    features = ['filePath', 'rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr', 'mfcc']
    features_data = []
    count = 1
    for audioFile in df_raw_dataset['filePath']:
        print('sample-file nr.: ' + str(count))
        features_data.append(get_features(audioFile))
        count = count + 1

    df_song_features = pd.DataFrame(features_data, columns=features)
    df_features_dataset = df_raw_dataset.merge(df_song_features, on='filePath')

    del df_raw_dataset
    del df_song_features

    return df_features_dataset


# load audio-file
# calculate features
# return feature array
def get_features(filePath):
    features_data = []
    y, sr = librosa.load(filePath)

    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return [filePath, np.mean(rmse), np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
            np.mean(zcr), np.mean(mfcc)]

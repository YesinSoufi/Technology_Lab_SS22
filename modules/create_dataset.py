# %%
from random import sample
from modules import cluster_dataset, feature_extraction, createSamples
import os
import pandas as pd
import sys

# %%
# variables
toCutAudioPath = os.path.abspath('../AudioData/AudioData.wav')
sampleSavePath = '../AudioData/AudioDataSamples/'
sampleLength = 0.02
cluster = 3
if not os.path.exists(toCutAudioPath):
    print("Fix code")
if not os.path.exists(sampleSavePath):
    print("Fix code")

# %%
# create samples from cutting one long track
df_dataset_samples = createSamples.createSamples(myAudioPath=toCutAudioPath, savePath=sampleSavePath, sampleLength=sampleLength)
df_dataset_samples.to_csv('../dataset_csv/dataset_samples_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_samples

# %%
# extract features from AudioData.wav samples
df_load_samples = pd.read_csv('../dataset_csv/dataset_samples_' + str(sampleLength) + '_seconds.csv')
df_dataset_features = feature_extraction.extract_features(df_load_samples)
df_dataset_features.to_csv('../dataset_csv/dataset_features_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_features

# %%
# cluster features and append label column
df_load_features = pd.read_csv('../dataset_csv/dataset_features_' + str(sampleLength) + '_seconds.csv')
df_dataset_labels = cluster_dataset.cluster_data(df_load_features, cluster = cluster)
df_dataset_labels.to_csv('../dataset_csv/dataset_labels_' + str(sampleLength) + '_seconds_' + str(cluster) + '_cluster.csv', index=False)
df_dataset_labels

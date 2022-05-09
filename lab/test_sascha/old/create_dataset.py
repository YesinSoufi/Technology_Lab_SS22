# %%
import createSamples
import feature_extraction
import cluster_dataset
import os
import pandas as pd
import numpy as np
from pathlib import Path


# %%
# create samples from audiodata.wav with 10 sec length
toCutAudioPath = '../../AudioData/AudioData.wav'
sampleSavePath = '../../AudioData/AudioDataSamples/'
sampleLength = 10

df_dataset_samples = createSamples.createSamples(myAudioPath=toCutAudioPath, savePath=sampleSavePath, sampleLength=sampleLength)
df_dataset_samples.to_csv('process_exports/dataset_samples.csv', index=False)
df_dataset_samples


# %%
# extract features from AudioData.wav samples
df_load_samples = pd.read_csv('process_exports/dataset_samples.csv', index_col=[0])
df_dataset_features = feature_extraction.extract_features(df_load_samples)
df_dataset_features.to_csv('process_exports/dataset_features.csv', index=False)
df_dataset_features

# %%
# cluster features and append label column
cluster = 2000

#df_load_features = pd.read_csv('process_exports/dataset_features.csv', index_col=[0])
df_load_features = pd.read_csv(r'C:\Users\sasch\Documents\GitHub\Technology_Lab_SS22\lab\test_sascha\process_exports\dataset_features.csv', index_col=[0])

df_dataset_labels = cluster_dataset.cluster_data(df_load_features, cluster = cluster)
#df_dataset_labels.to_csv('process_exports/dataset_labels.csv')
#df_dataset_labels

# %%

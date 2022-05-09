# %%
from random import sample
import createSamples
import feature_extraction
import cluster_dataset
import pandas as pd

# %%
# variables
toCutAudioPath = '/Users/OKaplan/Documents/GitHub/Technology_Lab_SS22_3/AudioData/Classical/Philip Ravenel - The Last Day Of November.wav'
sampleSavePath = '/Users/OKaplan/Documents/GitHub/Technology_Lab_SS22_3/AudioData/Classical/AudioDataSamples/withLabel/'
sampleLength = 6
#cluster = 1000

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
#df_dataset_labels = cluster_dataset.cluster_data(df_load_features, cluster = cluster)
#df_dataset_labels.to_csv('../dataset_csv/dataset_labels_' + str(sampleLength) + '_seconds_' + str(cluster) + '_cluster.csv', index=False)
#df_dataset_labels

# %%

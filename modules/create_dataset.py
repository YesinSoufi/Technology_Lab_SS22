# %%
from random import sample
import createSamples
import feature_extraction
import cluster_dataset
import pandas as pd



# %%
# variables AudioData\AudioData.wav
toCutAudioPath = r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\AudioData\AudioData.wav'
sampleSavePath = r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\AudioData\AudioDataSamples'
sampleLength = 0.02
cluster = 10

# %%
# create samples from cutting one long track
df_dataset_samples = createSamples.createSamples(toCutAudioPath, sampleSavePath,
                                                 sampleLength)
df_dataset_samples.to_csv(r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\dataset_csv\dataset_samples_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_samples

# %%
# extract features from AudioData.wav samples
df_load_samples = pd.read_csv(r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\dataset_csv\dataset_samples_' + str(sampleLength) + '_seconds.csv')
df_dataset_features = feature_extraction.extract_features(df_load_samples)
df_dataset_features.to_csv('../dataset_csv/dataset_features_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_features

# %%
# cluster features and append label column
df_load_features = pd.read_csv(r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\dataset_csv\dataset_features_' + str(sampleLength) + '_seconds.csv')
df_dataset_labels = cluster_dataset.cluster_data(df_load_features, cluster=cluster)
df_dataset_labels.to_csv(
    r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\dataset_csv\dataset_labels_' + str(sampleLength) + '_seconds_' + str(cluster) + '_cluster.csv', index=False)
df_dataset_labels

# %%

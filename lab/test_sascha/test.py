# %%
import createSamples

myAudioPath = '../../AudioData/AudioData.wav'
savePath = '../../AudioData/AudioDataSamples/'
sampleLength = 10


df_Test = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_Test

# %%
import pandas as pd

df_csv = pd.read_csv('sample_csv/sample_features_df.csv')

import cluster_dataset

df_labeldata = cluster_dataset.cluster_data(df_csv)
df_labeldata


# %%

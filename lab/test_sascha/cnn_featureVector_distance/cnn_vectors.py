#%%
import createSamples
import createSpectrograms
import pandas as pd
import librosa
import createBatchDataset

#%%
#cut samples
#myAudioPath = 'Audio/Metre-Slide.wav'
myAudioPath = 'Audio/Metre-Waves.wav'
#savePath = 'Audio/Samples/'
savePath = 'Audio/Samples_2/'

sampleLength = 6

df_dataset_samples = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_dataset_samples.to_csv('Audio/CSV/dataset_samples2_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_samples

#%%
#load samples
df_samples = pd.read_csv('Audio/CSV/dataset_samples2_6_seconds.csv')
df_samples

#%%
#create spectrograms
nameID = 10
for row in df_samples.itertuples():
    if row.ID != len(df_samples.index):
        clip_start, sr_start = librosa.load(row.filePath, mono=True, duration=1.001)
        clip_end, sr_end = librosa.load(row.filePath, mono=True, offset=4.999)
        savePath = 'Audio/Spectrograms_2/'
        startName =  'Start/ST' + str(nameID) + '.png'
        endName = 'End/EN' + str(nameID) + '.png'
        createSpectrograms.create_spectrogram(clip = clip_start,sample_rate = sr_start, save_path = savePath + startName)
        createSpectrograms.create_spectrogram(clip = clip_end,sample_rate = sr_end, save_path = savePath + endName)
    nameID = nameID + 10

# %%
import createBatchDataset
import tensorflow_datasets as tfds

specPath_start = 'Audio/Spectrograms/Start'
specPath_end = 'Audio/Spectrograms/End'

df_start = tfds.as_dataframe(createBatchDataset.loadSpectograms(specPath_start))
df_end = tfds.as_dataframe(createBatchDataset.loadSpectograms(specPath_end))

#%%
df_start.to_csv('Audio/CSV/Tensorflo_Dataset_Spectograms_Start1.csv')
df_end.to_csv('Audio(CSV/Tensorflow_Dataset_Spectograms_End1.csv')

# %%

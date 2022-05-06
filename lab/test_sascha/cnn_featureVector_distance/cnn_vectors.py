#%%
import createSamples
import createSpectrograms
import pandas as pd
import librosa

#%%
myAudioPath = 'Audio/Metre-Slide.wav'
savePath = 'Audio/Samples/'
sampleLength = 6

df_dataset_samples = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_dataset_samples.to_csv('Audio/CSV/dataset_samples_' + str(sampleLength) + '_seconds.csv', index=False)
df_dataset_samples

#%%
df_samples = pd.read_csv('Audio/CSV/dataset_samples_6_seconds.csv')
df_samples

#%%
for row in df_samples.itertuples():
    if row.ID != len(df_samples.index):
        clip_start, sr_start = librosa.load(row.filePath, mono=True, duration=1.001)
        clip_end, sr_end = librosa.load(row.filePath, mono=True, offset=4.999)
        savePath = 'Audio/Spectrograms/'
        startName =  'Start/' + str(row.ID) + '.png'
        endName = 'End/' + str(row.ID) + '.png'
        createSpectrograms.create_spectrogram(clip = clip_start,sample_rate = sr_start, save_path = savePath + startName)
        createSpectrograms.create_spectrogram(clip = clip_end,sample_rate = sr_end, save_path = savePath + endName)


# %%


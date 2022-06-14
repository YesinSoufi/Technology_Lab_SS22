#%%
import AudioUtil
import pandas as pd
from pathlib import Path
from natsort import natsort_keygen

#%%
#create 3 sec samples from all audio files
#6 Songs training
#2 Songs generate music

musicPath = 'C:/Users/sasch/Music/Techlab_Music/full_songs'
savePath = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/'
sampleLength = 3

for file in Path(musicPath).glob('*.wav'):
    saveDir = savePath + file.stem + '/'
    AudioUtil.cutSamples(file, saveDir, sampleLength, file.stem)
    print('Finished song: ' + file.name)


# %%
#create csv from samples with waveform column and sample name

audioDir = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/twirlingTwins'
data = []

for file in Path(audioDir).glob('*.wav'):
    name = file.stem
    waveform, sr = AudioUtil.loadWaveform(file)
    data.append([str(name), waveform.tolist()])

df_samples = pd.DataFrame(data, columns=['name', 'waveform'])

df_samples = df_samples.sort_values(
    by="name",
    key=natsort_keygen()
)

df_samples.reset_index(drop=True, inplace=True)

df_samples.to_csv('C:/Users/sasch/Music/Techlab_Music/samples_3sec/twirlingTwins.csv', index=False)


# %%
#test import csv
from ast import literal_eval
import numpy as np

csv = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/burble.csv'
df_test = pd.read_csv(csv)

# %%

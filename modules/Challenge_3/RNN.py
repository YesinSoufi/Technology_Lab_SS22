#%%
import AudioUtil
import pandas as pd
from pathlib import Path
from natsort import natsort_keygen
from ast import literal_eval
import numpy as np

#%%
#create 3 sec samples from all audio files
#6 Songs training
#2 Songs generate music

musicPath = 'C:/Users/Sascha/Music/Techlab_Music/full_songs'
savePath = 'C:/Users/Sascha/Music/Techlab_Music/samples_3sec/'
sampleLength = 3

for file in Path(musicPath).glob('*.wav'):
    saveDir = savePath + file.stem + '/'
    AudioUtil.cutSamples(file, saveDir, sampleLength, file.stem)
    print('Finished song: ' + file.name)


# %%
#create csv from samples with waveform column and sample name

audioDir = 'C:/Users/Sascha/Music/Techlab_Music/samples_3sec/'
data = []

songname = 'TwirlingTwins'
print(songname)

for file in Path(audioDir+songname).glob('*.wav'):
    name = file.stem
    waveform, sr = AudioUtil.loadWaveform(file)
    data.append([str(name), waveform.tolist()])

df_samples = pd.DataFrame(data, columns=['name', 'waveform'])

df_samples = df_samples.sort_values(
    by="name",
    key=natsort_keygen()
)

df_samples.reset_index(drop=True, inplace=True)

print('export samples csv')
df_samples.to_csv('C:/Users/Sascha/Music/Techlab_Music/samples_3sec/' + songname + '.csv', index=False)

#create labeled matching samples
csv = 'C:/Users/Sascha/Music/Techlab_Music/samples_3sec/' + songname + '.csv'
df_samples = pd.read_csv(csv)
df_samples['waveform'] = df_samples['waveform'].apply(literal_eval)
df_samples

df_match_samples_1 = pd.DataFrame(columns = ['training_waveform'])
df_match_samples_1['training_waveform'] = df_samples['waveform'] + df_samples['waveform'].shift(-1)
df_match_samples_1['label'] = 1
df_match_samples_1 = df_match_samples_1[:-1]

df_match_samples_2 = pd.DataFrame(columns = ['training_waveform'])
df_match_samples_2['training_waveform'] = df_samples['waveform'] + df_samples['waveform'].shift(-2)
df_match_samples_2['label'] = 1
df_match_samples_2 = df_match_samples_2[:-2]

print('export matching pairs csv')
df_match_samples_1.to_csv('C:/Users/Sascha/Music/Techlab_Music/samples_3sec/training_samples/' + songname + '_next_match.csv', index=False)
print('export second matching pairs csv')  
df_match_samples_2.to_csv('C:/Users/Sascha/Music/Techlab_Music/samples_3sec/training_samples/' + songname + '_next2_match.csv', index=False)

# %%
#create labeled wrong samples
name = 'TwirlingTwins'

print('load file')
#csv = 'C:/Users/Sascha/Music/Techlab_Music/samples_3sec/' + name + '.csv'
csv = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/' + name + '.csv'
df_samples = pd.read_csv(csv)
df_samples['waveform'] = df_samples['waveform'].apply(literal_eval)

print('create non matching pairs')
df_wrong_match = pd.DataFrame(columns=['name', 'training_waveform', 'label'])
for row in df_samples.itertuples():
    index = 13
    for _ in range(5):
        training_waveform = row.waveform + df_samples.iloc[index]['waveform']
        new_row = {'name':row.name, 'training_waveform': training_waveform, 'label':0}
        df_wrong_match = df_wrong_match.append(new_row, ignore_index=True)
        index = index + 1

print(name)
print('export non-matching pairs csv')  
#df_wrong_match.to_csv('C:/Users/Sascha/Music/Techlab_Music/samples_3sec/training_samples/' + name + '_wrong_match.csv', index=False)
df_wrong_match.to_csv('C:/Users/sasch/Music/Techlab_Music/samples_3sec/training_samples/' + name + '_wrong_match.csv', index=False)

#%%
df_match_wrong = pd.DataFrame(columns = ['training_waveform'])
#df_match_wrong['training_waveform'] = df_samples['waveform'] + df_samples['waveform'].shift(-20)
df_match_wrong['training_waveform'] = df_samples['waveform'] + np.roll(df_samples['waveform'], 20)
df_match_wrong['label'] = 0
df_match_wrong = df_match_wrong[:18]

df_match_wrong
# %%
df_match_wrong['training_waveform'][2]
# %%

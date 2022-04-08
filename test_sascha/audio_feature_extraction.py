# %%
import os
from pyexpat import features
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import librosa.display
import sklearn
import matplotlib.pyplot as plt

# %%
for file in Path('sample_music').glob('*.mp3'):
    print(os.path.basename(file))

# %%
data = []

for file in Path('sample_music').glob('*.mp3'):
    data.append([os.path.basename(file), file])

df = pd.DataFrame(data, columns=['name', 'filePath'])

df
# %%

test_audio = df.iloc[0,1]
x , sr = librosa.load(test_audio)

plt.figure(figsize=(20, 5))
librosa.display.waveshow(x, sr=sr)

# %%
test_audio_spec = librosa.stft(x)
test_audio_spec_dezibel = librosa.amplitude_to_db(abs(test_audio_spec))
plt.figure(figsize=(20, 5))
librosa.display.specshow(test_audio_spec_dezibel, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# %%
librosa.display.specshow(test_audio_spec_dezibel, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# %%
features_data = []
for file in df['filePath']:
    audio, sr2 = librosa.load(file)
    zero_crossing = librosa.zero_crossings(audio, pad=False)
    features_data.append(sum(zero_crossing))

df['zero_crossing'] = features_data
df
# %%
print(sum(df.iloc[0,2]))


# %%
df = df.rename(columns={'name':'audio_name', 'zero_crossing':'ZCR'})
df
# %%
df['ID'] = df.index+1
df
# %%
df = df[['ID','audio_name', 'filePath', 'ZCR']]
df

# %%
df.to_csv('sample_csv/zcr_df.csv', index=False)
# %%
features_data = []
for file in df['filePath']:
    y, sr2 = librosa.load(file)

    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr2)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr2)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr2)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr2)
    zcr = librosa.feature.zero_crossing_rate(y)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features_data.append([np.mean(rmse),np.mean(chroma_stft), np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff), np.mean(zcr)])

features = ['rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr']
df_features = pd.DataFrame(features_data, columns=features)
df_features

# %%
df = df.join(df_features)
df
# %%
df.to_csv('sample_csv/sample_features_df.csv', index=False)
# %%

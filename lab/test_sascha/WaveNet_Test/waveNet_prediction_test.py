#%%
import scipy.io
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment

#%%
def loadSample(samplePath):
    print('Load Sample: ' + str(samplePath))
    sampleRate, audioBuffer = scipy.io.wavfile.read(samplePath)
    audioArray = np.mean(audioBuffer, axis=1)

    duration = len(audioBuffer)/sampleRate
    time = np.arange(0,duration,1/sampleRate)

    #audio = AudioSegment.from_wav(samplePath)
    #audio = audio.set_channels(1)

    #plt.plot(time, audio)
    #plt.xlabel('Time [s]')
    #plt.ylabel('Amplitude')
    #plt.title('TEST')
    #plt.show()

    return sampleRate, audioArray

#%%
df_samples = pd.read_csv('../../../dataset_csv/dataset_samples_4_seconds.csv', index_col='ID')
df_samples['sampleRate'] = np.nan
#df_samples['audioArray'] = df_samples['audioArray'].astype(object)

row = 0
audioData = []
for wavPath in df_samples['filePath']:
    sampleRate, audioArray = loadSample('..\\..\\' + wavPath)
    df_samples.iloc[row, 2] = sampleRate
    audioData.append(audioArray)
    row = row + 1

df_samples

# %%
samplePath = '..\\..\\' + df_samples.iloc[20,1]
loadSample(samplePath)

# %%
#df = pd.DataFrame(audioData)
df = pd.DataFrame({"audioData": audioData})
df

# %%
audioData
# %%
result = pd.merge(
    df_samples,
    df,
    how='left',
    left_index=True, # Merge on both indexes, since right only has 0...
    right_index=True # all the other rows will be NaN
)
# %%
result
# %%
result.columns = result.columns.droplevel(0)
result
# %%
result.to_csv('audioData_test.csv')
# %%

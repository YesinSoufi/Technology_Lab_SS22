# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import mir_eval 
import IPython.display
import urllib
from pathlib import Path

plt.rcParams['figure.figsize'] = (20, 6)

# %%
filenamePath = r'C:\Users\sasch\Documents\GitHub\Technology_Lab_SS22\create_dataset_sascha\sample_music\Metre_Lite.mp3'

# %%
sound = IPython.display.Audio(filenamePath)
sound

# %%
x, fs = librosa.load(filenamePath)
print(fs)

# %%
librosa.display.waveshow(x, fs, sr=22050)

# %%
onset_frames = librosa.onset.onset_detect(x, sr=fs, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=fs)
onset_samples = librosa.frames_to_samples(onset_frames)

# %%
x_with_beeps = mir_eval.sonify.clicks(onset_times, fs, length=len(x))
IPython.display.Audio(x + x_with_beeps, rate=fs)


# %%
def extract_features(x, fs):
    zcr = librosa.zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]

# %%
frame_sz = fs*0.090
features = np.array([extract_features(x[i:i+frame_sz], fs) for i in onset_samples])
print(features.shape)


# %%

# %%
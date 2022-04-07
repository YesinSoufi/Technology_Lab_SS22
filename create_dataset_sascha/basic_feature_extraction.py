# %%
'exec(%matplotlib inline)'

from email.mime import audio
from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import urllib
import IPython.display as ipd
import librosa
import librosa.display



# %%
audio_examples = []
for p in Path('sample_music').glob('*.mp3'):
    audio_examples.append(librosa.load(p))
    print('inside loop')

# %%
audio = [
    librosa.load(p)[0] for p in Path('sample_music').glob('*.mp3')
]
len(audio)

# %%
len(audio_examples)
print(audio_examples[0])

# %%
plt.figure(figsize=(20, 6))
for i, x in enumerate(audio):
    x = np.asarray(x).astype(np.float32)
    plt.subplot(2, 5, i+1)
    librosa.display.waveshow(x[:10000])
    plt.ylim(-1, 1)
# %%
print(audio[0])
# %%
def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]

# %%
audio_features = np.array([extract_features(x) for x in audio])
# %%
print(audio_features[0])
# %%
plt.figure(figsize=(14, 5))
plt.hist(audio_features[:,0], color='b', range=(0, 0.2), alpha=0.5, bins=20)
plt.legend('audio')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Count')
# %%
plt.figure(figsize=(14, 5))
plt.hist(audio_features[:,1], color='b', range=(0, 4000), bins=30, alpha=0.6)
plt.legend('audio')
plt.xlabel('Spectral Centroid (frequency bin)')
plt.ylabel('Count')
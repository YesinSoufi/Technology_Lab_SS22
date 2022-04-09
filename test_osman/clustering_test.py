# %%
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa.display, mir_eval, IPython.display, urllib.request
plt.rcParams['figure.figsize'] = (14, 4)

# %%

y,sr = librosa.load('test_audio/Metre_Lite.mp3')



# %%
print(sr)

# %%

librosa.display.waveshow(y, sr)

# %%

onset_frames = librosa.onset.onset_detect(y, sr=sr, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_samples = librosa.frames_to_samples(onset_frames)

# %%

y_with_beeps = mir_eval.sonify.clicks(onset_times, sr, length=len(y))
IPython.display.Audio(y + y_with_beeps, rate=sr)

# %%
def extract_features(y, sr):
    zcr = librosa.zero_crossings(y).sum()
    energy = scipy.linalg.norm(y)
    return [zcr, energy]

# %%

frame_sz = sr
features = numpy.array([extract_features(y[i:i+frame_sz], sr) for i in onset_samples])
print (features.shape)

# %%

min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)
print (features_scaled.shape)
print (features_scaled.min(axis=0))
print (features_scaled.max(axis=0))
# %%

plt.scatter(features_scaled[:,0], features_scaled[:,1])
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Spectral Centroid (scaled)')

# %%


model = sklearn.cluster.KMeans(n_clusters=2)
labels = model.fit_predict(features_scaled)
print (labels)
# %%

plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))

# %%

y_with_beeps = mir_eval.sonify.clicks(onset_times[labels==0], sr, length=len(y))
IPython.display.Audio(y + y_with_beeps, rate=sr)
# %%

y_with_beeps = mir_eval.sonify.clicks(onset_times[labels==1], sr, length=len(y))
IPython.display.Audio(y + y_with_beeps, rate=sr)
# %%

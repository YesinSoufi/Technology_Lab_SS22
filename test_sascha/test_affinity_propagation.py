# %%
import numpy
import scipy
import matplotlib.pyplot as plt
import sklearn
import librosa 
import IPython.display

plt.rcParams['figure.figsize'] = (14, 4)

#%%
#x, sr = librosa.load('sample_music/Metre_Lite.mp3')
x, sr = librosa.load('sample_music/Metre_High.mp3')
print(type(x))

#%%
onset_frames = librosa.onset.onset_detect(x, sr=sr, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_samples = librosa.frames_to_samples(onset_frames)

# %%
def extract_features(x, fs):
    zcr = librosa.zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]

# %%
frame_sz = sr
features = numpy.array([extract_features(x[i:i+frame_sz], sr) for i in onset_samples])
print(features.shape)

# %%
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)
print(features_scaled.shape)
print(features_scaled.min(axis=0))
print(features_scaled.max(axis=0))

# %%
plt.scatter(features_scaled[:,0], features_scaled[:,1])
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Spectral Centroid (scaled)')

# %%
model = sklearn.cluster.AffinityPropagation()
labels = model.fit_predict(features_scaled)
print(labels)

# %%
print(type(features_scaled))
print(features_scaled)

# %%
model = sklearn.cluster.KMeans(n_clusters=2)
labels = model.fit_predict(features_scaled)
print(labels)

# %%
plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==2,1], c='r')
#plt.scatter(features_scaled[labels==2,0], features_scaled[labels==3,1], c='y')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1', 'Class 2'))



# %%
plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))
# %%
print(features_scaled)

# %%

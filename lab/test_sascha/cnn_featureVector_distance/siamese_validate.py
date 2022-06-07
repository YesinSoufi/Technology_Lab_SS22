#%%
from mpl_toolkits.mplot3d import Axes3D
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, pickle
import sklearn, pandas as pd, seaborn as sn
from keras.models import Model, load_model, Sequential
from keras import backend as K
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load models
model = load_model(os.getcwd()+"/color_encoder.h5")
siamese_model = load_model(os.getcwd()+"/color_siamese_model.h5")

# Load test data
f = open(os.getcwd()+"/test_images.pkl", 'rb')
test_end_im, test_start_im = pickle.load(f)
f.close()

# Read files
names = list(test_end_im) + list(test_start_im)
names1 = [x for x in names if 'EN' in x]
names2 = [x for x in names if 'ST' in x]

test_im = []
for i in range(len(names)) :
    path = names[i].replace('\\', '/')
    path = path.replace('ers/sasch', 'ers/Sascha')
    test_im.append(cv2.imread(path))


#%%
r,c,_ = test_im[0].shape
test_im = np.array(test_im)
test_im = test_im.reshape((len(test_im), r,c,3))
names = [x.split("/")[-1] for x in names]

test_im = 1 - test_im/255


#%%
# Predict
pred = model.predict(test_im)

#%%
num = int(pred.shape[0]/2)
colors = ['red', 'green'] # set colors of target labels

# Set target labels
y = [colors[0] for i in range(num)]
y += [colors[1] for i in range(num)]

feat1 = pred[:,0]
feat2 = pred[:,1]

# Plot 3d scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(feat1, feat2, c=y, marker='.')
plt.show()
# %%
pred.shape

# %%
pred[0]
# %%

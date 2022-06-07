#%%
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils.vis_utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix
from natsort import os_sorted

import warnings
warnings.filterwarnings('ignore')

#%%
# Import color encoder which uses siamese networks
from siamese_test_model import train_color_encoder


# Prepare data for different shapes but same colors

#%%
dir = os.getcwd()+"/Audio/Spectrograms"

images = []
y_col = []
end_spec = []
start_spec = []

for root, dirs, files in os.walk(dir, topdown=False):
    for name in os_sorted(files):
        fullname = os.path.join(root, name)
        if fullname.find(".png") != -1 :
            images.append(fullname)
            if fullname.find("EN") != -1 :
                y_col.append(0)
                end_spec.append(0)
            elif fullname.find("ST") != -1 :
                y_col.append(1)
                start_spec.append(1)

# 0 = EN-Spectrograms
# 1 = ST-Spectrograms
y_col = np.array(y_col)
images = np.array(images)

# %%
# Generate positive samples
end_im = images[np.where(y_col==0)]
start_im = images[np.where(y_col==1)]

# Test images
test_end_im = end_im[50:]
test_start_im = start_im[50:]

# Read only 20 images from each class for training
train_end_im = end_im[:20]
train_start_im = start_im[:20]

#%%
positive_end_start = list(zip(train_end_im, train_start_im[1:]))
#positive_blue = list(itertools.combinations(blue_im, 2))

# %%
# Generate negative samples
dir = os.getcwd()+"/Audio/Spectrograms_2"

images_neg = []
y_col_neg = []
end_spec_neg = []
start_spec_neg = []

for root, dirs, files in os.walk(dir, topdown=False):
    for name in os_sorted(files):
        fullname = os.path.join(root, name)
        if fullname.find(".png") != -1 :
            images_neg.append(fullname)
            if fullname.find("EN") != -1 :
                y_col_neg.append(0)
                end_spec_neg.append(0)
            elif fullname.find("ST") != -1 :
                y_col_neg.append(1)
                start_spec_neg.append(1)

# 0 = EN-Spectrograms
# 1 = ST-Spectrograms
y_col_neg = np.array(y_col_neg)
images_neg = np.array(images_neg)

end_im_neg = images_neg[np.where(y_col==0)]
start_im_neg = images_neg[np.where(y_col==1)]

# Test images
test_end_im_neg = end_im_neg[50:]
test_start_im_neg = start_im_neg[50:]

# Read only 20 images from each class for training
train_end_im_neg = end_im_neg[:20]
train_start_im_neg = start_im_neg[:20]

# %%
negative_end_start = list(zip(train_end_im, train_start_im_neg[1:]))
# %%

# Create pairs of images and set target label for them. Target output is 1 if pair of images have same color else it is 0.
end_X1 = []
start_X2 = []
same_y = []

#zusammengehörende end + start
for fname in positive_end_start:
    im = cv2.imread(fname[0])
    end_X1.append(im)
    im = cv2.imread(fname[1])
    start_X2.append(im)
    same_y.append(1)

#end und start aus verschiedenen songs
for fname in negative_end_start :
    im = cv2.imread(fname[0])
    end_X1.append(im)
    im = cv2.imread(fname[1])
    start_X2.append(im)
    same_y.append(0)

same_y = np.array(same_y)
end_X1 = np.array(end_X1)
start_X2 = np.array(start_X2)
end_X1 = end_X1.reshape((len(negative_end_start) + len(positive_end_start), 217, 223, 3))
start_X2 = start_X2.reshape((len(negative_end_start) + len(positive_end_start), 217, 223, 3))

end_X1 = 1 - end_X1/255
start_X2 = 1 - start_X2/255

print("Color data : ", end_X1.shape, start_X2.shape, same_y.shape)

#%%
# Save test data
f = open(os.getcwd()+"/test_images.pkl", 'wb')
pickle.dump([test_end_im, test_start_im], f)
f.close()

# %%
# train model
train_color_encoder(end_X1, start_X2, same_y)
# %%

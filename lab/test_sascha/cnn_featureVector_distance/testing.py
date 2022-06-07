#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2


# %%
specfilelist=os.listdir('Audio/Spectrograms/Start')
specfilelist=['Audio/Spectrograms/Start'+filename for filename in specfilelist]
specfilelist

# %%
start = tf.keras.preprocessing.image_dataset_from_directory(
  'Audio/Spectrograms/Start',
  label_mode=None,
  batch_size=70
)

end = tf.keras.preprocessing.image_dataset_from_directory(
  'Audio/Spectrograms/Start',
  label_mode=None,
  batch_size=70
)


# %%
start = np.array([cv2.imread(f.path) / 255 for f in os.scandir("Audio/Spectrograms/Start")])
end = np.array([cv2.imread(f.path) / 255 for f in os.scandir("Audio/Spectrograms/End")])

train_start = start[:205]
validate_start = start[205:]
train_end = end[:205]
validate_end = end[205:]

#%%
train_start.shape

# %%
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Flatten(),  
      layers.Dense(48391, activation='sigmoid'),
      layers.Reshape((217, 223))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)


#%%
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# %%
autoencoder.fit(train_start, train_start,
                epochs=10,
                shuffle=True,
                validation_data=(validate_start, validate_start))

#%%

batch_size = 10
img_height = 223
img_width = 217


val_ds = tf.keras.utils.image_dataset_from_directory(
  'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/lab/test_sascha/cnn_featureVector_distance/Audio/Spectrograms/Start',
  #validation_split=0.2,
  #subset="validation",
  label_mode=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images in val_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")

# %%
for image_batch in val_ds:
  print(image_batch.shape)


# %%
val_ds
# %%

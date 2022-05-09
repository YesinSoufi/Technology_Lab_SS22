import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 10
img_height = 223
img_width = 217

def loadSpectograms(folderPath):
    spectrograms_ds = tf.keras.utils.image_dataset_from_directory(
        folderPath,
        label_mode=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return spectrograms_ds

def showSpectograms(dataset):
    plt.figure(figsize=(10, 10))
    for images in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Cropping2D, Conv1D, Reshape, MaxPooling1D, Dense,Dropout,Activation,Flatten, Conv2D, MaxPooling2D, MaxPool2D, Conv2DTranspose, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses
from datetime import datetime

def cRNN_Prototyp():
    model = Sequential()

    model.add(Conv1D(256, 3, activation='relu', input_shape=(132300,1)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    return model
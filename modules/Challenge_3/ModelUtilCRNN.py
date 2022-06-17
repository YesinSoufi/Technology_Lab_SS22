from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense,Dropout,Flatten


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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense,Dropout,Flatten, LSTM


def cRNN_Prototyp():
    model = Sequential()

    model.add(Conv1D(64, 3, activation='relu', input_shape=(44100,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(2))

    #model.add(Flatten())
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model
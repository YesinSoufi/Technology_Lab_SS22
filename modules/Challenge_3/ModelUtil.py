
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense,Dropout,Activation,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from datetime import datetime

def predictSimilarity(sample_array, model):
    prediction = model.predict(sample_array)
    return prediction

def trainModel(epochs, batch_size, model, data):

    checkpointer = ModelCheckpoint(filepath='saved_models/checkpoints/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(data, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    return model

#old models/ random models
def firstModel(num_labels):
    model=Sequential()
    ###first layer
    #model.add(Dense(100,input_shape=(176400,)))
    model.add(Dense(100,input_shape=(22050,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###fourth layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###fifth layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    ###final layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    
    return model

def cnnModel():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(22050,)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(24, activation='softmax'))

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#looking into group 4 model
def model_Gruppe4(num_layer):
    model = Sequential()

    model.add(Conv1D(256, 3, activation='relu', input_shape=(22050,1)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_layer))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


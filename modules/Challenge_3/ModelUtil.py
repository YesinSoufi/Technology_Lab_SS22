from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Cropping2D, Conv1D, Reshape, MaxPooling1D, Dense,Dropout,Activation,Flatten, Conv2D, MaxPooling2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses
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

#-----------------------------------#
#   NN-Models
#-----------------------------------#
def cnnTest():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', 
                    input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling2D((2, 2), name='maxpool_3'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), name='maxpool_4'))
    model.add(Flatten())

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def autoEncoderTest():
    conv_encoder = Sequential([
        #Reshape([217,334, 3], input_shape=[217, 334, 3]),
        #Reshape([220,332, 3], input_shape=[256, 332, 3]),
        #Conv2D(16, kernel_size=3, padding="SAME", activation="selu", input_shape=(256,256,3)),
        #MaxPool2D(pool_size=2),
        #Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
        #MaxPool2D(pool_size=2),
        #Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
        #MaxPool2D(pool_size=2)  
        #Input(shape=(256, 256, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256,256,3)),
        Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)      
    ])

    #print(conv_encoder.layers[5].output_shape)
    #print(conv_encoder.layers[6].output_shape)

    conv_decoder = Sequential([
        #Conv2DTranspose(32, kernel_size=3, padding="SAME", activation="selu",
        #                            input_shape=[32, 32, 64]),
        #Conv2DTranspose(16, kernel_size=3, padding="SAME", activation="selu"),
        #Conv2DTranspose(3, kernel_size=3, padding="SAME", activation="sigmoid"),
        #Reshape([217,334], input_shape=[220, 332, 1])
        Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])

    conv_ae = Sequential([conv_encoder, conv_decoder])

    #conv_ae.compile(optimizer='adam', loss = losses.MeanSquaredError())
    conv_ae.compile(optimizer='adam', loss="binary_crossentropy")
    

    return conv_ae, conv_encoder, conv_decoder

def autoEncoder1():
    conv_encoder = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256,256,3)),
        Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(4, (3,3), activation='relu', padding='same', strides=2)
    ])

    conv_decoder = Sequential([
        Conv2DTranspose(4, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])

    conv_ae = Sequential([conv_encoder, conv_decoder])

    conv_ae.compile(optimizer='adam', loss="binary_crossentropy")
    

    return conv_ae, conv_encoder, conv_decoder

def autoEncoder2():
    conv_encoder = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256,256,3)),
        Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(4, (3,3), activation='relu', padding='same', strides=2),
        Flatten()
    ])

    conv_decoder = Sequential([
        Reshape([8,8,4]),
        Conv2DTranspose(4, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])

    conv_ae = Sequential([conv_encoder, conv_decoder])

    opt = Adam(learning_rate=0.0015)
    conv_ae.compile(optimizer=opt, loss="binary_crossentropy")
    #conv_ae.compile(optimizer='adam', loss="binary_crossentropy")
    

    return conv_ae, conv_encoder, conv_decoder

# ------------------------------------------------ #
#old models/ random models
# ------------------------------------------------ #
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


#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

#%%

x_train = pd.read_csv("test.csv")
y_train = pd.read_csv("test.csv") #Labels

y_train = y_train == 0

x_test = pd.read_csv("test.csv")
y_test = pd.read.csv("test.csv")

y_test = y_test == 0

y_train = print(to_categorical(y_train)) 
y_test = print(to_categorical(y_test)) 


#%%
# First CNN Model
model = Sequential

model.add(Conv1D(30, kernal_size= 3, activation= "relu", input_shape = (784, )))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25)) #nur dann verwenden, wenn zwischen Trainings- und Testdaten ein riesen Unterschied besteht
model.add(Dense(10, activation= "softmax")) # 10 Ausgänge in unserem Neuronalen Netz

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=["accuracy"]) #categorial Loss weil wir mehrere Ausgänge haben

model.fit(
    x_train.reshape(60000, 784), 
    y_train,
    epochs=80,
    batch_size=1000)

#%%
#Test CNN-LSTM Ansatz
model = Sequential

model.add(Conv1D(30, kernal_size= 3, activation= "relu", input_shape = (784,)))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25)) #nur dann verwenden, wenn zwischen Trainings- und Testdaten ein riesen Unterschied besteht
model.add(LSTM(20, activation = "relu"))
model.add(Dense(10, activation= "softmax")) # 10 Ausgänge in unserem Neuronalen Netz

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=["accuracy"]) #categorial Loss weil wir mehrere Ausgänge haben

model.fit(
    x_train.reshape(60000, 784), 
    y_train,
    epochs=80,
    batch_size=1000)

#%%
#Second CNN Model 
model = Sequential

model.add(Conv1D(30, kernal_size= 3, activation= "relu", input_shape = (784, )))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Dense(50, activation= "sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(10, activation= "softmax"))

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(
    x_train.reshape(60000, 784), 
    y_train,
    epochs=80,
    batch_size=1000)



#%%
#Third CNN Model 
model = Sequential

model.add(Conv1D(30, kernal_size= 5, activation= "relu", input_shape = (784, )))
model.add(Dense(50, activation= "sigmoid"))
model.add(Dense(10, activation= "softmax"))

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(
    x_train.reshape(60000, 784), 
    y_train,
    epochs=80,
    batch_size=1000)


#%%

model.evaluate(x_test.reshape(-1, 784), y_test) #reshape 10.000 weil wir im Testdaten nur 10.000 Testdaten haben

#%%
x_train[0].reshape(1, 784) # 1 Datensatz mit 784 Pixel (siehe oben input_shape)

#%%

model.predict(x_train[0].reshape(1, 784))

#%%

y_train_pred = model.predict(x_train.reshape(60000, 784))

#%%
#Werte runden um eine 0 oder 1 Wahrscheinlichkeit zu bekommen

np.round(y_train_pred).reshape(-1) #Numpy weiß, dass 60.000 Daten drin sind deshalb reicht auch eine -1 als Parameter aus

#%%

np.mean(np.around(y_train_pred).reshape(-1) == y_train) #np.mean berechnet den Durschnitt der Genauigkeit 
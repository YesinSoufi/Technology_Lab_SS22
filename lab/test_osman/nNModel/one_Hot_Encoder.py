#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
# Model defination
model = Sequential

model.add(Dense(100 , activation= "relu", input_shape = (784, ))) #Hidden Layer
model.add(Dense(10, activation= "softmax")) # 10 Ausgänge in unserem Neuronalen Netz

model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics=["accuracy"]) #categorial Loss weil wir mehrere Ausgänge haben

#%%

x_train.reshape(60000, 784)

#%%
#Train Model

model.fit(
    x_train.reshape(60000, 784), 
    y_train,
    epochs=10,
    batch_size=1000
    )


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
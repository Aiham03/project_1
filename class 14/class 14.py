import pandas as pd
from keras import sequential()
from keras.layers import InputLayer, Dense

df = pd.read_csv('diabetes.csv')

model = sequential()
model.add(InputLayer(input_shape=(12,1)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

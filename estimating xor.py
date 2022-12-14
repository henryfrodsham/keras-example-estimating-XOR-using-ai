import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation
A = np.array([[0,0],[1,0],[0,1],[1,1]],"float32")    #training data
B = np.array([[0],[1],[1],[0]],"float32") #data to ccompare 

model = Sequential()
model.add(Dense(16,input_dim = 2,activation = 'relu')) #hidden layer
model.add(Dense(1,activation = 'sigmoid')) #output layer
model.compile(loss = 'mse', optimizer = 'adam')
#modify the amount of output layer neurons to affect the output of the model, for instance 2 neurons will output 2 values

history = model.fit(A,B, epochs=2000,batch_size = 4, verbose = 0)
#verbose shows output if set to 1
#epochs is the amount of traning cycles are to be completed 


print(model.predict(A))
# will output a float value from 0-1, in this case a value higher than 0.5 is considered true and below is false

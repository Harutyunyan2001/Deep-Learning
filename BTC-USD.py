import yfinance as yf
import datetime 
import pandas as pd
import numpy as np
from keras.layers import *
from keras.models import Sequential,save_model,load_model
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import *
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = yf.Ticker("BTC-USD")
data = data.history(period="10y",interval="1d")
train = data.iloc[:,0:4]
test = data.iloc[:,3:4]

train = train.to_numpy()
test = test.to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

time_step = 15
x,y = [],[]

for i in range(len(train)-time_step):
    row = [a for a in train[i:i+time_step]]
    x.append(row)
    lable = [a for a in test[i+time_step]]
    y.append(lable)
x = np.array(x)
y = np.array(y) 

train_len = int(len(x)*0.9)
x_train, y_train = x[:train_len], y[:train_len]
x_test, y_test = x[train_len:], y[train_len:]
y_train = y_train.reshape(int(len(y_train)),-1)
y_test = y_test.reshape(int(len(y_test)),-1)

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (time_step,4)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['mae'])
model.fit(x_train,y_train,epochs=200,batch_size=100)  #epochs=100,batch_size=100
model.summary()

pred_x_test = model.predict(x_test)
pred_x_test = scaler.inverse_transform(pred_x_test)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

y_train_len = list(range(len(y_train)))
y_test_len = list(range(len(y_test)))
y_test_len = list(range(len(y_train),len(y_test)+len(y_train)))

plt.plot(y_train_len,y_train.flatten())
plt.plot(y_test_len,y_test.flatten())
plt.plot(y_test_len,pred_x_test,color = "r")
plt.show()


test_minus_pred = np.subtract(y_test,pred_x_test)
print(test_minus_pred)
print(len(test_minus_pred))
test_minus_pred_mean = np.sum(test_minus_pred)/len(test_minus_pred)
print("mijin_shexum",test_minus_pred_mean)

x_new = train[len(train)-time_step:]
x_new = np.expand_dims(x_new,axis=0)
x_new_pred = model.predict(x_new)
x_new_pred = scaler.inverse_transform(x_new_pred)

x_new_pred_mean = x_new_pred+(test_minus_pred_mean)

if y_test[-1] > x_new_pred:
    print("sell")
else:print("buy")

if y_test[-1] > x_new_pred_mean:
    print("sell")
else:print("buy")



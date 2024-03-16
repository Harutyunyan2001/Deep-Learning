import yfinance as yf
import datetime 
import pandas as pd
import numpy as np
from keras.layers import *
from keras.models import Sequential,save_model,load_model
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


scaler = MinMaxScaler(feature_range=(0,1))

amazon = yf.Ticker("AMZN")
end_data = datetime.datetime.now().strftime("%Y-%m-%d")
data = amazon.history(period="2y",interval="1h")
train = data.iloc[:,0:4]
test = data.iloc[:,3:4]

train = train.to_numpy()
test = test.to_numpy()
test1 = test.reshape(1,-1)
#train = scaler.fit_transform(train)
#test = scaler.fit_transform(test)

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
model.compile(loss=MeanSquaredError(),optimizer=Adam(),metrics=RootMeanSquaredError())
model.fit(x_train,y_train,epochs=100,batch_size=10)
model.summary()

y_train_len = list(range(len(y_train)))
y_test_len = list(range(len(y_test)))
y_test_len = list(range(len(y_train),len(y_test)+len(y_train)))


last15 = train[len(train)-15]
last14 = train[len(train)-14]
last13 = train[len(train)-13]
last12 = train[len(train)-12]
last11 = train[len(train)-11]
last10 = train[len(train)-10]
last9 = train[len(train)-9]
last8 = train[len(train)-8]
last7 = train[len(train)-7]
last6 = train[len(train)-6]
last5 = train[len(train)-5]
last4 = train[len(train)-4]
last3 = train[len(train)-3]
last2 = train[len(train)-2]
last1 = train[-1]
x_new = np.concatenate([[last15],[last14],[last13],[last12],[last11],[last10],[last9],[last8],[last7],[last6],[last5],[last4],[last3],[last2],[last1]])


plt.plot(y_train_len,y_train.flatten())
plt.plot(y_test_len,y_test.flatten())
plt.plot(y_test_len,model.predict(x_test),color = "r")
plt.show()

x_test = x_test[-1]
x_test = np.expand_dims(x_test, axis=0)
x_test = model.predict(x_test)


print("data",test[-1])
print("test", x_test)
print("tarb",test[-1] - x_test )
print("data2-data1",test[len(test)-2]-test[-1])
print("data2-test",test[len(test)-2]-x_test)







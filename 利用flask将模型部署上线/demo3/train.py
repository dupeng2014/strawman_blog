import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence

seq = np.array([i/float(10) for i in range(10)])

# 5个样本、2个时间步长和1个特征
X = seq.reshape(2, 5, 1)
y = seq.reshape(2, 5)

# define LSTM configuration
n_neurons = 5
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(5, 1)))
model.add(Dense(5))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
model.save('./model/model.h5')


# x_test = np.array([i/float(5) for i in range(5)])
# x_test = x_test.reshape(-1, 5, 1)
# result = model.predict(x_test, batch_size=n_batch, verbose=0)
# print(result)
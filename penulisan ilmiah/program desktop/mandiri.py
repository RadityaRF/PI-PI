import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

#loading train data
X_df = pd.read_csv('dataset\BMRI.JK.csv')
X_values = X_df.iloc[:3000,-1].values
X_train = X_values.reshape(-1, 1)

#scalling the train data
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)

features_set = []
labels = []
for i in range(0, len(X_train)-20):
    features_set.append(X_train[i:i+20])
    labels.append(X_train[i+20])


features_set, labels = np.array(features_set), np.array(labels)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 100, batch_size = 32)

#loading test data

Y_df = pd.read_csv('dataset\BMRI.JK.csv')
Y_values = Y_df.iloc[3001:,-1].values
Y_test = Y_values.reshape(-1, 1)

Y_test = scaler.transform(Y_test)


test_features = []
true_labels = []
for i in range(0, len(Y_test)-20):
    test_features.append(Y_test[i:i+20])
    true_labels.append(Y_test[i+20])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], 20, 1))

predictions = model.predict(test_features)
predictions_unscaled = scaler.inverse_transform(predictions)
true_labels_unscaled = scaler.inverse_transform(true_labels)

#evaluations
plt.figure(figsize=(10,6))
plt.plot(true_labels_unscaled, color='blue', label='Harga Saham Asli')
plt.plot(predictions_unscaled , color='red', label='Harga Saham Prediksi')
plt.title('Prediksi Saham Bank Mandiri')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.show()
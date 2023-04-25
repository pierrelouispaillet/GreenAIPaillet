
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



filename = 'mnist_test.csv'
Data = pd.read_csv(filename)

Data_plot = pd.DataFrame([], columns=Data.columns)

for i in range(10):
    Data_plot = pd.concat([Data_plot, Data.loc[Data['label'] == i].iloc[0, :].to_frame().T], ignore_index=True)

Y_plot = np.array(Data_plot.pop('label'))
X_plot = np.array(Data_plot)
X_plot = X_plot.reshape([-1, 28, 28]).astype('float32')
X_plot /= 255

rows = 5
columns = 2
fig, ax = plt.subplots(rows, columns, figsize = [12, 6])
for i in range(rows):
    for j in range(columns):
        index = i * columns + j
        ax[i, j].imshow(X_plot[index, :, :], cmap = 'binary')
        ax[i, j].axis('on')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

X_plot_FFT = np.fft.fft2(X_plot)

X_plot_FFT_mag = np.log(np.abs(X_plot_FFT))
X_plot_FFT_mag -= np.mean(X_plot_FFT_mag)
X_plot_FFT_mag /= np.std(X_plot_FFT_mag)

X_plot_FFT_phase = np.angle(X_plot_FFT) / np.pi


rows = 5
columns = 4
fig, ax = plt.subplots(rows, columns, figsize = [12, 12])
for i in range(rows):
    for j in range(columns // 2):
        index = i * (columns // 2 ) + j
        ax[i, j*2].imshow(X_plot_FFT_mag[index, :, :], cmap = 'binary')
        ax[i, j*2].axis('on')
        ax[i, j*2].set_xticks([])
        ax[i, j*2].set_yticks([])
        ax[i, j*2+1].imshow(X_plot_FFT_phase[index, :, :], cmap = 'binary')
        ax[i, j*2+1].axis('on')
        ax[i, j*2+1].set_xticks([])
        ax[i, j*2+1].set_yticks([])



Y = np.array(Data.pop('label'))
X = np.array(Data).reshape([-1, 28, 28])

X_FFT = np.fft.fft2(X)

b = np.where(X_FFT != 0.0)
X_FFT_mag = np.zeros(X_FFT.shape)
X_FFT_mag[b] = np.log(np.abs(X_FFT[b]))
X_FFT_mag[X_FFT == 0.0] = -32.57791748631743

mean_FFTmag = np.nanmean(X_FFT_mag)
std_FFTmag = np.nanstd(X_FFT_mag)

X_FFT_mag -= mean_FFTmag
X_FFT_mag /= std_FFTmag


X_FFT_phase = np.angle(X_FFT) / np.pi

X = X.reshape([-1,28,28, 1])
X_FFT_mag = X_FFT_mag.reshape([-1,28,28, 1])
X_FFT_phase = X_FFT_phase.reshape([-1,28,28, 1])


X_3Ch = X

X_3Ch = np.append(X, X_FFT_mag, axis = 3)
X_3Ch = np.append(X_3Ch, X_FFT_phase, axis = 3)
X_3Ch.shape

from sklearn.model_selection import train_test_split, StratifiedKFold

X_3Ch_train, X_3Ch_test, Y_train, Y_test = train_test_split(X_3Ch, Y, test_size = 0.2, shuffle = True)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

drop_out_rate = 0.1
num_class = 10

data_shape = X_3Ch.shape[1:]

def build_model(data_shape, num_class, drop_out_rate):

    CNN = Sequential([
        Conv2D(32, (4,4), activation = 'relu', input_shape=data_shape),
        MaxPooling2D(pool_size=(2, 2), strides = 1, padding = 'same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2), strides = 1, padding = 'same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'),
        Dropout(drop_out_rate),
        Flatten(),
        Dense(8192, activation = 'relu'),
        Dropout(drop_out_rate),
        Dense(2048, activation = 'relu'),
        Dropout(drop_out_rate),
        Dense(512, activation = 'relu'),
        Dropout(drop_out_rate),
        Dense(128, activation = 'relu'),
        Dropout(drop_out_rate),
        Dense(64, activation = 'relu'),
        Dropout(drop_out_rate),
        Dense(num_class, activation = 'softmax'),
    ])
    
    CNN.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    return CNN

callback = EarlyStopping(patience=10, restore_best_weights = True)

num_epochs = 20

history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy  = []


skf = StratifiedKFold(n_splits = 5)

count = 1

for train_index, test_index in skf.split(X_3Ch_train, Y_train):
    
    print('traning k-fold:', count)
    
    CNN = build_model(data_shape, num_class, drop_out_rate)

    history = CNN.fit(X_3Ch_train[train_index],
            Y_train[train_index],
            batch_size = 512,
            epochs = num_epochs,
            validation_data = (X_3Ch_train[test_index], Y_train[test_index]),
            verbose = 1)
    
    history_loss.append(history.history['loss'])
    history_accuracy.append(history.history['accuracy'])
    history_val_loss.append(history.history['val_loss'])
    history_val_accuracy.append(history.history['val_accuracy'])
    
    count += 1
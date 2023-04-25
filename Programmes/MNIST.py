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

from sklearn.model_selection import train_test_split, StratifiedKFold

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

filename = 'mnist_test.csv'
Data = pd.read_csv(filename)

Y = np.array(Data.pop('label'))
X = np.array(Data).reshape([-1, 28, 28, 1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

drop_out_rate = 0.1
num_class = 10
data_shape = X_train.shape[1:]

def build_model(data_shape, num_class, drop_out_rate):
    CNN = Sequential([
        Conv2D(32, (4,4), activation='relu', input_shape=data_shape),
        MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(drop_out_rate),
        Conv2D(32, (4,4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(drop_out_rate),
        Flatten(),
        Dense(8192, activation='relu'),
        Dropout(drop_out_rate),
        Dense(2048, activation='relu'),
        Dropout(drop_out_rate),
        Dense(512, activation='relu'),
        Dropout(drop_out_rate),
        Dense(128, activation='relu'),
        Dropout(drop_out_rate),
        Dense(64, activation='relu'),
        Dropout(drop_out_rate),
        Dense(num_class, activation='softmax'),
    ])

    CNN.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    return CNN

callback = EarlyStopping(patience=10, restore_best_weights=True)

num_epochs = 20

history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []

skf = StratifiedKFold(n_splits=5)

count = 1

for train_index, test_index in skf.split(X_train, Y_train):
    print('training k-fold:', count)

    CNN = build_model(data_shape, num_class, drop_out_rate)

    history = CNN.fit(X_train[train_index],
            Y_train[train_index],
            batch_size=512,
            epochs=num_epochs,
            validation_data=(X_train[test_index], Y_train[test_index]),
            verbose=1)

    history_loss.append(history.history['loss'])
    history_accuracy.append(history.history['accuracy'])
    history_val_loss.append(history.history['val_loss'])
    history_val_accuracy.append(history.history['val_accuracy'])

    count += 1

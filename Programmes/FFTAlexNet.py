import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = to_categorical(train_labels, 10), to_categorical(test_labels, 10)



class FourierConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', activation=None, **kwargs):
        super(FourierConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(input_shape[-1], self.filters, *self.kernel_size), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        inputs_fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))

        # Expand kernel dimensions to match input dimensions
        kernel_expanded = tf.expand_dims(tf.expand_dims(self.kernel, 0), 0)
        kernel_fft = tf.signal.fft2d(tf.cast(kernel_expanded, tf.complex64))

        output_fft = tf.multiply(inputs_fft, kernel_fft)
        outputs = tf.signal.ifft2d(output_fft)
        outputs = tf.abs(outputs)

        if self.padding == 'SAME':
            pad_top = (self.kernel_size[0] - 1) // 2
            pad_bottom = self.kernel_size[0] - 1 - pad_top
            pad_left = (self.kernel_size[1] - 1) // 2
            pad_right = self.kernel_size[1] - 1 - pad_left
            outputs = tf.pad(outputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs



# Create AlexNet model
def create_alexnet():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), implementation = 0))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', implementation = 0))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', implementation = 0))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))




    return model


def create_fourier_alexnet():
    model = Sequential()
    model.add(FourierConv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(FourierConv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(FourierConv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model


# Compile and train the model
model = create_alexnet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=1, validation_split=0.1)



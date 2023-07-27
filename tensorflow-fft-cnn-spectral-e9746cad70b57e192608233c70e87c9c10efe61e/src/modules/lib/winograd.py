import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical

from layers import *



def CNN():

    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = datasets.cifar10.load_data()
    cifar_train_images, cifar_test_images = cifar_train_images / 255.0, cifar_test_images / 255.0
    cifar_train_labels, cifar_test_labels = to_categorical(cifar_train_labels), to_categorical(cifar_test_labels)

    class WinogradConv2D(layers.Layer):
        def __init__(self, filters, kernel_size, **kwargs):
            super(WinogradConv2D, self).__init__(**kwargs)
            self.filters = filters
            self.kernel_size = kernel_size

        def build(self, input_shape):
            self.kernel = self.add_weight("kernel", (self.kernel_size, self.kernel_size, input_shape[-1], self.filters))

        def call(self, inputs):
            # Implement the Winograd algorithm here
            # You can refer to this paper for more details: https://arxiv.org/abs/1509.09308
            # Note that implementing the full Winograd algorithm is beyond the scope of this example
            # We will use the standard 2D convolution as a placeholder
            return tf.nn.conv2d(inputs, self.kernel, strides=(1, 1), padding="SAME")

    # Create a CNN model using the custom WinogradConv2D layer
    pool1= layers.max_pooling2d(
                inputs=conv1,
                pool_size=3,
                strides=2,
                padding='SAME',
                name='max_pool_1'
            )

    model = models.Sequential([
        #WinogradConv2D(32, 3, input_shape=(32, 32, 3), name="winograd_conv1"),
        spectral_conv_layer(
				input_x=pool1,
				in_channel=32,
				out_channel=32,
				kernel_size=3,
				random_seed=0,
				m=3
			),
        spectral_conv_layer(
                        input_x=pool1,
                        in_channel=64,
                        out_channel=256,
                        kernel_size=3,
                        random_seed=0,
                        m=3
                    ),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.summary()

    # Compile and train the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(cifar_train_images, cifar_train_labels, epochs=4, validation_data=(cifar_test_images, cifar_test_labels))

CNN()


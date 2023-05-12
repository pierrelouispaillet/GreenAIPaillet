import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import DeepSaki

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model architecture with the adjusted input shape
model = DeepSaki.models.ResNet(inputShape=(32,32,3), number_of_levels=3, filters=64, useResidualIdentityBlock=True, 
                        residual_cardinality=1, final_activation="softmax")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, callbacks=[earlystopping])

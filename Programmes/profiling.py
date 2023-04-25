import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np
import datetime
from datetime import datetime
from packaging import version
import os


device_name = tf.test.gpu_device_name()

(ds_train, ds_test), ds_info = tfds.load("mnist", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True)

)
# Create a TensorBoard callback
logs = “logs/” + datetime.now().strftime(“%Y%m%d-%H%M%S”)tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1, profile_batch = ‘500,520’)model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks = [tboard_callback])
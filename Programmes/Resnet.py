import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import datasets,models,layers

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplDramDomain
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.device.rapl_device import RaplUncoreDomain
from pyJoules.device import Device

# Set the number of cores you want to use
num_cores = 8

# Create a configuration object
config = tf.compat.v1.ConfigProto()

# Set the number of inter- and intra-operation threads
config.inter_op_parallelism_threads = num_cores
config.intra_op_parallelism_threads = num_cores

# Set the GPU options, if applicable
config.gpu_options.allow_growth = True

# Create a session with the custom configuration
session = tf.compat.v1.Session(config=config)

# Set the session as the default for TensorFlow
tf.compat.v1.keras.backend.set_session(session)

Accuracy=[]
 
csv_handler = CSVHandler('resultRes.csv')

@measure_energy(handler=csv_handler, domains=[RaplPackageDomain(0), NvidiaGPUDomain(0), RaplDramDomain(0), RaplCoreDomain(0)])
def Res(i,j):
    from keras.datasets import cifar10
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train).toarray()
    Y_test = encoder.transform(Y_test).toarray()
    Y_val =  encoder.transform(Y_val).toarray()

    from keras.preprocessing.image import ImageDataGenerator
    aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                                height_shift_range=0.05)
    aug.fit(X_train)

    n_elements = X_train.shape[0]
    end_index = int(np.floor(n_elements * i))
    X_train = X_train[:end_index]
    Y_train = Y_train[:end_index]



    from keras.callbacks import EarlyStopping
    from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
    from keras.models import Sequential
    from keras.models import Model
    import tensorflow as tf

    

    
    class ResnetBlock(Model):
        """
        A standard resnet block.
        """

        def __init__(self, channels: int, down_sample=False):
            """
            channels: same as number of convolution kernels
            """
            super().__init__()

            self.__channels = channels
            self.__down_sample = down_sample
            self.__strides = [2, 1] if down_sample else [1, 1]

            KERNEL_SIZE = (3, 3)
            # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
            INIT_SCHEME = "he_normal"

            self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                                kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_1 = BatchNormalization()
            self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                                kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_2 = BatchNormalization()
            self.merge = Add()

            if self.__down_sample:
                # perform down sampling using stride of 2, according to [1].
                self.res_conv = Conv2D(
                    self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
                self.res_bn = BatchNormalization()

        def call(self, inputs):
            res = inputs

            x = self.conv_1(inputs)
            x = self.bn_1(x)
            x = tf.nn.relu(x)
            x = self.conv_2(x)
            x = self.bn_2(x)

            if self.__down_sample:
                res = self.res_conv(res)
                res = self.res_bn(res)

            # if not perform down sample, then add a shortcut directly
            x = self.merge([x, res])
            out = tf.nn.relu(x)
            return out


    class ResNet18(Model):

        def __init__(self, num_classes, **kwargs):
            """
                num_classes: number of classes in specific classification task.
            """
            super().__init__(**kwargs)
            self.conv_1 = Conv2D(64, (7, 7), strides=2,
                                padding="same", kernel_initializer="he_normal")
            self.init_bn = BatchNormalization()
            self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
            self.res_1_1 = ResnetBlock(64)
            self.res_1_2 = ResnetBlock(64)
            self.res_2_1 = ResnetBlock(128, down_sample=True)
            self.res_2_2 = ResnetBlock(128)
            self.res_3_1 = ResnetBlock(256, down_sample=True)
            self.res_3_2 = ResnetBlock(256)
            self.res_4_1 = ResnetBlock(512, down_sample=True)
            self.res_4_2 = ResnetBlock(512)
            self.avg_pool = GlobalAveragePooling2D()
            self.flat = Flatten()
            self.fc = Dense(num_classes, activation="softmax")

        def call(self, inputs):
            out = self.conv_1(inputs)
            out = self.init_bn(out)
            out = tf.nn.relu(out)
            out = self.pool_2(out)
            for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
                out = res_block(out)
            out = self.avg_pool(out)
            out = self.flat(out)
            out = self.fc(out)
            return out
        

    model = ResNet18(10)
    model.build(input_shape = (None,32,32,3))
    #use categorical_crossentropy since the label is one-hot encoded
    from keras.optimizers import SGD
    # opt = SGD(learning_rate=0.1,momentum=0.9,decay = 1e-04) #parameters suggested by He [1]
    model.compile(optimizer = "adam",loss='categorical_crossentropy', metrics=["accuracy"]) 
    model.summary()

    from keras.callbacks import EarlyStopping

    es = EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_acc")
    #I did not use cross validation, so the validate performance is not accurate.
    STEPS = len(X_train) / 256
    history = model.fit(aug.flow(X_train,Y_train,batch_size = 256), steps_per_epoch=STEPS, batch_size = 256, epochs=j, validation_data=(X_train, Y_train),callbacks=[es])



    # list all data in history
    print(history.history.keys())
    

    ## Evaluation

    ModelLoss, ModelAccuracy = model.evaluate(X_train, Y_train)

    print('Model Loss is {}'.format(ModelLoss))
    print('Model Accuracy is {}'.format(ModelAccuracy))
    print('The number of element is : ')
    n_elements = X_train.shape[0]
    print(n_elements)

    Accuracy.append(ModelAccuracy)

for i in [0.2,0.4,0.6,0.8,1]:
    for j in [10,15,20,25]:
    	Res(i,j)


#Res(1,1)

print(Accuracy)
csv_handler.save_data()
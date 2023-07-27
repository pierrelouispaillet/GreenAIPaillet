import tensorflow 
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
import time
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
import keras.backend



from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplDramDomain
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.device.rapl_device import RaplUncoreDomain
from pyJoules.device import Device


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


import energyusage

	


np.random.seed(1000)

plt.style.use('fivethirtyeight')
start_time = time.time()

Accuracy=[]

csv_handler = CSVHandler('Alexresult.csv')

@measure_energy(handler=csv_handler, domains=[RaplPackageDomain(0), NvidiaGPUDomain(0), RaplDramDomain(0), RaplCoreDomain(0)])
def AlexNet(i,j):
#Instantiation
	AlexNet = Sequential()

	#1st Convolutional Layer
	AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#2nd Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#3rd Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))

	#4th Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
        

	#5th Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#Passing it to a Fully Connected layer
	AlexNet.add(Flatten())
	# 1st Fully Connected Layer
	AlexNet.add(Dense(4096*i, input_shape=(32,32,3,)))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	AlexNet.add(Dropout(0.4))

	#2nd Fully Connected Layer
	AlexNet.add(Dense(4096*i))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	#Add Dropout
	AlexNet.add(Dropout(0.4))

	#3rd Fully Connected Layer
	AlexNet.add(Dense(1000*i))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	#Add Dropout
	AlexNet.add(Dropout(0.4))

	#Output Layer
	AlexNet.add(Dense(j))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('softmax'))

	#Model Summary
	AlexNet.summary()

	AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])

	#Keras library for CIFAR dataset
	from keras.datasets import cifar10
	(x_train, y_train),(x_test, y_test)=cifar10.load_data()
	if j ==9:
		classes_to_exclude = [9]
	if j ==8:
		classes_to_exclude = [8,9]
	if j ==7:
		classes_to_exclude = [7,8,9]
	if j ==6:
		classes_to_exclude = [6,7,8,9]
	if j ==5:
		classes_to_exclude = [5,6,7,8,9]
		

	mask_train = np.isin(y_train, classes_to_exclude, invert=True)
	mask_test = np.isin(y_test, classes_to_exclude, invert=True)
	x_train, y_train = x_train[mask_train.squeeze()], y_train[mask_train.squeeze()]
	x_test, y_test = x_test[mask_test.squeeze()], y_test[mask_test.squeeze()]

	#Train-validation-test split
	from sklearn.model_selection import train_test_split
	x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.3)

	

	#Since we have 10 classes we should expect the shape[1] of y_train,y_val and y_test to change from 1 to 10
	y_train=to_categorical(y_train)
	y_val=to_categorical(y_val)
	y_test=to_categorical(y_test)

	

	#Image Data Augmentation
	from keras.preprocessing.image import ImageDataGenerator

	train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1 )

	val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)

	test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True,zoom_range=.1)

	#Fitting the augmentation defined above to the data
	train_generator.fit(x_train)
	val_generator.fit(x_val)
	test_generator.fit(x_test)

	#Learning Rate Annealer
	from keras.callbacks import ReduceLROnPlateau
	lrr= ReduceLROnPlateau(   monitor='val_accuracy',   factor=.01,   patience=3,  min_lr=1e-5) 

	#Defining the parameters
	batch_size= 32
	epochs= 15
	train_size = i  # The percentage of the training set to be used during training (0.0 - 1.0) 
    
	 # apply the train_size of the trainset

	n_elements = x_train.shape[0]
	end_index = int(np.floor(n_elements * train_size))
	x_train = x_train[:end_index]
	y_train = y_train[:end_index]

	#Training the model
	AlexNet.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = x_train.shape[0]//batch_size, validation_data = val_generator.flow(x_val, y_val, batch_size=batch_size), validation_steps = 250, callbacks = [lrr], verbose=1)

	print("\nCalculating the accuracy over the testset:")
	accuracy = AlexNet.evaluate(x_test, y_test, verbose=1) 

	Accuracy.append(accuracy)
    
    


for i in [0.2,0.4,0.6,0.8,1]:
    for j in [9,8,7,6,5]:
    	AlexNet(i,j)



#energyusage.evaluate(AlexNet, 1 , 10, pdf = True)

csv_handler.save_data()

print(Accuracy)
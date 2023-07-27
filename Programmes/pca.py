
# acquire MNIST data through Keras API
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplDramDomain
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.device.rapl_device import RaplUncoreDomain
from pyJoules.device import Device
import pandas as pd


csv_handler = CSVHandler('pca.csv')

H = []

@measure_energy(handler=csv_handler, domains=[RaplPackageDomain(0), NvidiaGPUDomain(0), RaplDramDomain(0), RaplCoreDomain(0)])
def PCAM(i,l):

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # reshape (flatten) data before PCA


    train_images = np.reshape(train_images, (-1, 784))
    test_images = np.reshape(test_images, (-1, 784))



    # compute the number of elements to keep
    n_elements = train_images.shape[0]
    end_index = int(np.floor(n_elements * i)) 

    # slice the arrays
    train_images = train_images[:end_index]
    train_labels = train_labels[:end_index]

    # normalize data before PCA
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # apply PCA once to
    # select the best number of components


    pca_784 = PCA(n_components=784)
    pca_784.fit(train_images)



    # apply PCA again with 100 components
    # about 90% of the variability retained
    # transformation is applied to both
    # train and test sets
    pca_100 = PCA(n_components=l)
    pca_100.fit(train_images)
    train_images_reduced = pca_100.transform(train_images)
    test_images_reduced = pca_100.transform(test_images)

    # verify shape after PCA
    print("Train images shape:", train_images_reduced.shape)
    print("Test images shape: ", test_images_reduced.shape)

    # get exact variability retained
    print("\nVar retained (%):", 
        np.sum(pca_100.explained_variance_ratio_ * 100))

    # convert labels to a one-hot vector


    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # define network architecture


    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(l, ))) # input layer
    MLP.add(Dense(64, activation='relu')) # hidden layer 1
    MLP.add(Dense(32, activation='relu')) # hidden layer 2
    MLP.add(Dense(10, activation='softmax')) # output layer

    # optimization
    MLP.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # train (fit)
    history = MLP.fit(train_images_reduced, train_labels, 
                    epochs=20, batch_size=128, verbose=1,
                    validation_split=0.15)

    # evaluate performance on test data
    test_loss, test_acc = MLP.evaluate(test_images_reduced, test_labels,
                                            batch_size=128,
                                            verbose=0)
    
    H.append(test_acc)

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # or save to csv: 
    hist_csv_file = 'history' + str(i) + str(l) + '.csv' 
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)




for l in [0.2,0.4,0.6,0.8,1]:
    for i in [100,150,250,300,350]:  
        PCAM(l,i)
csv_handler.save_data()
print(H)
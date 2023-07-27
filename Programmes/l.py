import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplDramDomain
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.device.rapl_device import RaplUncoreDomain
from pyJoules.device import Device

csv_handler = CSVHandler('resultL.csv')

@measure_energy(handler=csv_handler, domains=[RaplPackageDomain(0), NvidiaGPUDomain(0), RaplDramDomain(0), RaplCoreDomain(0)])
def model():
    # Téléchargement et préparation du jeu de données CIFAR10
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalisation des valeurs de pixel pour être entre 0 et 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Création du modèle CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Ajout des couches entièrement connectées avec la régularisation L1
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Compilation et entraînement du modèle
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
model()

csv_handler.save_data()
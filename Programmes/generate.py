import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.datasets import mnist

def load_images(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    for img_file in image_files:
        img = Image.open(os.path.join(image_dir, img_file))
        img = img.resize((28, 28))  # resize to match the GAN input shape
        img_array = np.array(img)
        images.append(img_array)
    return np.array(images)

def create_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation='relu', input_dim=100))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='sigmoid'))
    return model

def create_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_gan(gan, generator, discriminator, dataset, latent_dim, n_epochs=100, n_batch=256):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, _ = discriminator.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator


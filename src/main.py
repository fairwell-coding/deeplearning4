from typing import Tuple

import tensorflow as tf
from numpy.random import normal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, BatchNormalization, Reshape, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
import os

from tensorflow.python.keras.regularizers import L1L2

RANDOM_STATE = 42
DATA_PATH = "../data/"


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      #pls dont delete this


def __train_autoencoder(perturbed_data_set):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = perturbed_data_set

    # model, training_error = __model_14(x_val_transformed, x_train_transformed)
    model, training_error = __model_15(perturbed_data_set)
    testing_error = model.evaluate(x_test_perturb, x_test_transformed, batch_size=512)

    # Print the final training error, validation error and test accuracy
    print('Training loss: '"{:.2f}".format(training_error.history['loss'][-1]) + " Validation loss: " + "{:.2f}".format(training_error.history['val_loss'][-1]))
    print('Test loss: ' + "{:.4f}".format(testing_error))

    # Plot the evolution of the error during training
    plt.plot(training_error.history['loss'])
    plt.plot(training_error.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss', loc='top')
    plt.xlabel('epoch', loc='right')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.annotate('Test loss: ' + "{:.2f}".format(testing_error), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', color='red')
    plt.show()

    # Visual comparison of a reconstructed sample side by side with a clean sample
    sample_index = 100
    sample_for_prediction = x_val_perturb[sample_index].reshape((1, 28, 28, 1))
    pred = model.predict(sample_for_prediction)
    plt.figure()
    plt.imshow(x_val_transformed[sample_index])  # clean sample
    plt.figure()
    plt.imshow(pred.reshape((28, 28, 1)))  # perturb sample
    plt.show()


def __create_perturbed_data_set(load_kuzushiji=False):
    np.random.seed(RANDOM_STATE)

    if load_kuzushiji:
        with np.load(DATA_PATH + "x_train.npz", allow_pickle=True) as train_data:
            x_train = train_data[train_data.files[0]]

        with np.load(DATA_PATH + "y_train.npz", allow_pickle=True) as train_labels:
            y_train = train_labels[train_labels.files[0]]

        with np.load(DATA_PATH + "x_test.npz", allow_pickle=True) as test_data:
            x_test = test_data[test_data.files[0]]

        with np.load(DATA_PATH + "y_test.npz", allow_pickle=True) as test_labels:
            y_test = test_labels[test_labels.files[0]]

    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Convert to one-out-of-K encoding
    y_train_transformed = to_categorical(y_train, num_classes=10)
    y_test_transformed = to_categorical(y_test, num_classes=10)

    # Images are grayscale with size 28x28 -> reshape them that they only have a single color channel
    x_train_transformed = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test_transformed = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Normalize data, grayscale images have values in range 0-255 -> convert to 0-1
    x_train_transformed = (x_train_transformed.astype('float32')) / 255.0
    x_test_transformed = (x_test_transformed.astype('float32')) / 255.0
    x_train_transformed, x_val_transformed, y_train_transformed, y_val_transformed = train_test_split(x_train_transformed, y_train_transformed, test_size=0.2, random_state=RANDOM_STATE)

    x_train_perturb = __create_perturb_data(x_train_transformed)
    x_val_perturb = __create_perturb_data(x_val_transformed)
    x_test_perturb = __create_perturb_data(x_test_transformed)

    return x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb


def __create_perturb_data(data_set: np.ndarray):
    perturb_data = np.empty(data_set.shape)

    for index in range(len(data_set)):
        image = data_set[index].reshape(28, 28, 1)
        if index % 6 == 0:
            perturb_data[index] = __add_black_square_patch(image, int(np.random.uniform(3, 6)))
        if index % 6 == 1:
            perturb_data[index] = __change_brightness(image, brightness_change=np.random.uniform(-0.5, 0.5))
        if index % 6 == 2:
            perturb_data[index] = __rotate_image(image, max_angle=180)
        if index % 6 == 3:
            perturb_data[index] = __flip_image(image, vertical=True)
        if index % 6 == 4:
            perturb_data[index] = __flip_image(image, horizontal=True)
        if index % 6 == 5:
            perturb_data[index] = __add_gaussian_noise(image, 0.2)

    return perturb_data


def __add_black_square_patch(image: np.ndarray, size: int):
    """ Overlay a black square patch with given size at a random location over the image.

    :param image: image as 3dim-tensor
    :param size: size of black square (width & height)
    :return: modified image with black square patch as overlay
    """

    x_coordinate = int(np.floor(np.random.uniform(0, image.shape[0] + 1 - size)))
    y_coordinate = int(np.floor(np.random.uniform(0, image.shape[0] + 1 - size)))

    image[x_coordinate:x_coordinate + size, y_coordinate:y_coordinate + size] = 0.0  # overlay black square patch

    return image


def __add_gaussian_noise(image, stddev):
    """ Adds Gaussian noise to image by given standard deviation (possible range between 0 and 1, useful range between 0.5 and 0.01).

    :param image: image as 3dim-tensor
    :param stddev: standard deviation
    :return: modified image with added Gaussian noise
    """

    noise = np.random.normal(0.0, stddev, size=28 * 28).reshape(image.shape)
    return np.clip(image + noise, 0, 1)


def __change_brightness(image, brightness_change=None, stddev=None):
    """ Change brightness of image by either providing fixed normalized change or by drawing from a normal distribution.

    :param image: image as 3dim-tensor
    :param brightness_change: fixed brightness value of normalized image (useful range between -0.5 and 0.5)
    :param stddev: standard deviation for brightness (cut-off at 1)
    :return modified image with adjusted brightness
    """

    if not brightness_change:
        brightness_change = np.random.normal(0.0, stddev)

    return np.clip(image + brightness_change, 0, 1)


def __rotate_image(image, max_angle):
    """ Rotate image by angle drawn from normal distribution

    :param image: image as 3dim-tensor
    :param max_angle: maximum rotation angle
    :return rotateted image
    """
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = rotate(image, angle, reshape=False)
    return rotated


def __flip_image(image, horizontal=False, vertical=False):
    """ Flip image either by horizontal or vertical axis

    :param image: image as 3dim-tensor
    :param horizontal: perform horizontal flip
    :param vertical: perform vertical flip
    :return flipped image
    """
    if horizontal == True:
        flipped = np.flipud(image)
    elif vertical == True:
        flipped = np.fliplr(image)

    return flipped


# loss: 0.0787 - accuracy: 0.4784 - val_loss: 0.0800 - val_accuracy: 0.4651
def __base_model(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    # decoder
    model.add(Dense(4, activation='relu'))
    model.add(Reshape((2, 2, 1)))
    model.add(Conv2D(4, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((7, 7)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added more convolutional layers to base model
# loss: 0.0565 - accuracy: 0.4890 - val_loss: 0.0568 - val_accuracy: 0.4869
def __model_1(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    # decoder
    model.add(Dense(4, activation='relu'))
    model.add(Reshape((2, 2, 1)))
    model.add(Conv2D(4, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((7, 7)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added more dense layers to model_1, changed early stopping to val_loss which seems to work better
# loss: 0.0515 - accuracy: 0.4925 - val_loss: 0.0535 - val_accuracy: 0.4947
def __model_2(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # decoder
    model.add(Dense(4, activation='relu'))
    model.add(Reshape((2, 2, 1)))
    model.add(Conv2D(4, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((7, 7)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added another conv layer with normalization to model_2
# loss: 0.0877 - accuracy: 0.4893 - val_loss: 0.0882 - val_accuracy: 0.4942
def __model_3(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # decoder
    model.add(Dense(4, activation='relu'))
    model.add(Reshape((2, 2, 1)))
    model.add(Conv2D(4, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((7, 7)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 2 with Conv2DTranspose instead of Conv2D on the decoder
# loss: 0.0877 - accuracy: 0.4891 - val_loss: 0.0875 - val_accuracy: 0.4927
def __model_4(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # decoder
    model.add(Dense(4, activation='relu'))
    model.add(Reshape((2, 2, 1)))
    model.add(Conv2DTranspose(4, (2, 2), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((7, 7)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 5 first trained with original data, then with perturbed + some regularizers
# test accuracy: 0.6715, test error 0.09
# val accuracy: 0.6715, val loss: 0.0924
# train_accuracy: 0.6784, train loss: 0.0883
def __model_5(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu', kernel_regularizer=l2(l2=0.01)))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(49, activation='relu', kernel_regularizer=l2(l2=0.01)))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.01)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 6 first trained with original data, then with perturbed + some regularizers
# test accuracy: 0.6701, test error 0.12
# val accuracy: 0.6699, val loss: 0.1149
# train_accuracy: 0.6691, train loss: 0.1012
def __model_6(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu', ))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 7 first trained with original data, then with perturbed + some regularizers
# test accuracy: 0.6701, test error 0.12
# val accuracy: 0.6701, val loss: 0.1266
# train_accuracy: 0.6700, train loss: 0.0997
def __model_7(x_train_perturb, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu', ))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(5, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 8 just train with original data -> image should be reproduced well enough
# changed output activation to linear since this is kind of regression problem
# test accuracy: 0.6701, test error 0.12
# val accuracy: 0.6701, val loss: 0.1105
# train_accuracy: 0.6700, train loss: 0.1102
def __model_8(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu', ))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 9 just train with original data -> image should be reproduced well enough
# a liitle bit deeper than model 8, but it performs way better
# now one can recognise the image at the output of the model
# val accuracy: 0.6762, val loss: 0.0844
# train_accuracy: 0.6790, train loss: 0.0824
def __model_9(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 10 just train with original data -> image should be reproduced well enough
# this was too deep, predicted image is almost a quare
# val accuracy: 0.6701, val loss: 0.1224
# train_accuracy: 0.6784, train loss: 0.0830
def __model_10(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 11 just train with original data -> image should be reproduced well enough
# this was still too much -> output is just square
# val accuracy: 0.6701, val loss: 0.1191
# train_accuracy: 0.6703, train loss: 0.1048
def __model_11(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 12 just train with original data -> image should be reproduced well enough
# this better again -> output gets more similar
# val accuracy: 0.6728, val loss: 0.0896
# train_accuracy: 0.6748, train loss: 0.0853
def __model_12(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 13 just train with original data -> image should be reproduced well enough
# validation se loss just increased ->x doesn't work well
# val accuracy: 0.6701, val loss: 0.1413
# train_accuracy: 0.6696, train loss: 0.1116
def __model_13(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(16, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 14 just train with original data -> image should be reproduced well enough
# try out larger dense layers -> way better result than before
# val accuracy: 0.6884, val loss: 0.0702
# train_accuracy: 0.6883, train loss: 0.0700
def __model_14(x_val_transformed, x_train_transformed):
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(3, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_transformed, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


def __model_15(data: Tuple[np.ndarray]):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    # create model
    weight_regularizer = L1L2(0.0001)
    latent_space_dim = 4

    model = Sequential()

    # encoder
    model.add(Conv2D(filters=32,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                     kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                     # 3) instead of once (5, 5) is a modern efficient approach)
                     strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                     padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                     activation='relu',  # relu is considered a standard for conv layers and typically works really well
                     activity_regularizer=weight_regularizer,
                     kernel_initializer='he_uniform',  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
                     # 2)/num_inputs
                     input_shape=(28, 28, 1)))  # needs to be defined since it's the first layer of the network
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(normal(0.2, 0.05)))  # convolutional dropout using normal distribution with mean = 0.2 and std_deviation = 0.05
    model.add(Conv2D(filters=32,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                     kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                     # 3) instead of once (5, 5) is a modern efficient approach)
                     strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                     padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                     activation='relu',  # relu is considered a standard for conv layers and typically works really well
                     activity_regularizer=weight_regularizer,
                     kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(normal(0.2, 0.05)))  # convolutional dropout using normal distribution with mean = 0.2 and std_deviation = 0.05
    model.add(Flatten())  # flatten convolution output in order to feed into affine layers which learn the classification based on the convoluted learned feature representation in the conv layers
    model.add(Dense(128, activation='relu', activity_regularizer=weight_regularizer))  # dense layer to allow the network to learn the classification based on the learned convolutional feature
    # representation
    model.add(Dropout(0.5))  # typical dropout factor for dense layers

    # latent space
    model.add(Dense(latent_space_dim, activation='relu'))  # encoder output

    # decoder
    model.add(Dense(49, activation='relu', activity_regularizer=weight_regularizer))  # set output dimension suitable for reshape dimension
    model.add(Dropout(0.5))
    model.add(Reshape((7, 7, 1)))  # inverse operation of flatten in encoder
    model.add(UpSampling2D((2, 2)))  # inverse operation of MaxPooling2D in encoder
    model.add(Conv2DTranspose(filters=32,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                              kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                              # 3) instead of once (5, 5) is a modern efficient approach)
                              strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                              padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                              activation='relu',  # relu is considered a standard for conv layers and typically works really well
                              activity_regularizer=weight_regularizer,
                              kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(Dropout(normal(0.2, 0.05)))  # convolutional dropout using normal distribution with mean = 0.2 and std_deviation = 0.05
    model.add(UpSampling2D((2, 2)))  # inverse operation of MaxPooling2D in encoder
    model.add(Conv2DTranspose(filters=1,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                              kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                              # 3) instead of once (5, 5) is a modern efficient approach)
                              strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                              padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                              activation='relu',  # relu is considered a standard for conv layers and typically works really well
                              activity_regularizer=weight_regularizer,
                              kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(Dropout(normal(0.2, 0.05)))  # convolutional dropout using normal distribution with mean = 0.2 and std_deviation = 0.05

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


if __name__ == '__main__':
    perturbed_data_set = __create_perturbed_data_set(load_kuzushiji=True)  # set flag to true to load kuzushiji data
    __train_autoencoder(perturbed_data_set)

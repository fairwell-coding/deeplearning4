import pickle
from typing import Tuple

import tensorflow as tf
from numpy.random import normal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2, l1_l2, L1L2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, BatchNormalization, Reshape, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate


RANDOM_STATE = 42
DATA_PATH = "../data/"


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def __train_autoencoder(perturbed_data_set):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = perturbed_data_set

    model, training_error = __model_12(perturbed_data_set)
    
    testing_error = model.evaluate(x_test_perturb, x_test_transformed, batch_size=512)

    __save_model_weights_and_history(model, "12")

    # Print the final training error, validation error and test accuracy
    print('Training loss: '"{:.4f}".format(training_error.history['loss'][-1]) + " Validation loss: " + "{:.4f}".format(training_error.history['val_loss'][-1]))
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


def __create_comparison_images(data_set, index):
    """ Used to create image tuples
    """

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data_set

    model = load_model("final_trained_model")

    results = []
    for sample_perturb, sample_clean in zip(x_test_perturb[:index], x_test_transformed[:index]):
        eval_result = model.evaluate(sample_perturb.reshape((1, 28, 28, 1)), sample_clean.reshape((1, 28, 28, 1)))
        results.append(eval_result)

    worst_index = results.index(max(results))
    best_index = results.index(min(results))

    median_result = sorted(results)[int(len(results)/2)]
    median_index = results.index(median_result)

    # Visual comparison of a reconstructed sample side by side with clean samples
    for sample_index in [best_index, median_index, worst_index]:
        sample_for_prediction = x_val_perturb[sample_index].reshape((1, 28, 28, 1))
        pred = model.predict(sample_for_prediction)
        plt.figure()
        plt.imshow(x_val_transformed[sample_index])  # clean sample
        plt.figure()
        plt.imshow(pred.reshape((28, 28, 1)))  # perturb sample
        plt.show()



def __save_model_weights_and_history(model, model_number):
    model.save_weights('../checkpoints/model_{0}.h5'.format(model_number))
    with open('../checkpoints/model_{0}_hist'.format(model_number), 'wb') as file_pi:
        pickle.dump(model.history, file_pi)

def __load_model(model_path, data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    model = tf.keras.models.load_model(model_path)

    for index in range(10,20):
        sample_for_prediction = x_test_perturb[index].reshape((1, 28, 28, 1))
        pred = model.predict(sample_for_prediction)
        plt.figure()
        plt.imshow(x_test_transformed[index])  # clean sample
        plt.figure()
        plt.imshow(pred.reshape((28, 28, 1)))  # perturb sample
        plt.show()

def __create_perturbed_data_set():
    np.random.seed(RANDOM_STATE)

    with np.load(DATA_PATH + "x_train.npz", allow_pickle=True) as train_data:
        x_train = train_data[train_data.files[0]]

    with np.load(DATA_PATH + "y_train.npz", allow_pickle=True) as train_labels:
        y_train = train_labels[train_labels.files[0]]

    with np.load(DATA_PATH + "x_test.npz", allow_pickle=True) as test_data:
        x_test = test_data[test_data.files[0]]

    with np.load(DATA_PATH + "y_test.npz", allow_pickle=True) as test_labels:
        y_test = test_labels[test_labels.files[0]]

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


# Training loss: 0.10 Validation loss: 0.10 Test loss: 0.0997
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added more convolutional layers to base model
# Training loss: 0.09 Validation loss: 0.09 Test loss: 0.0946
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added more dense layers to model_1
# Training loss: 0.0908 Validation loss: 0.0917 Test loss: 0.0922
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Added another conv layer with normalization to model_2
# Training loss: 0.1090 Validation loss: 0.1091 Test loss: 0.1053
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 2 with Conv2DTranspose instead of Conv2D on the decoder
# Training loss: 0.1111 Validation loss: 0.1127 Test loss: 0.1080
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])
    return model, training_error


# Model 5 retrained
# val loss: 0.0865
# train loss: 0.0856
def __model_5(data):

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


# Model 6 retrained
# val loss: 0.1506
# train loss: 0.1158
def __model_6(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(3 , activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error

# Model 7
# val loss: 0.1561
# train loss: 0.1113
def __model_7(data):

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(7, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error

# Model 8
# val loss: 0.1089
# train loss: 0.1088
# test loss: 0.1053
def __model_8(data):

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(6, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error

# Model 8.1 -> check out different number of nodes for bottleneck layer
# val loss: 0.0886
# train loss: 0.0880
# test loss: 0.0914
def __model_8_1(data):

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(5, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error

# Model 9 retrained
#val loss: 0.1224
# train loss: 0.1091
def __model_9(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
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
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    pred = model.predict(x_val_transformed)
    plt.figure()
    plt.imshow(x_val_transformed[100])
    plt.figure()
    plt.imshow(pred[100])
    plt.show()

    return model, training_error


# Model 10
# val loss: 0.0837
#train loss: 0.0820
def __model_10(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


# Model 11
# val loss: 0.0830
# train loss: 0.0818
def __model_11(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


# Model 12 deepest network until now, works the best
# val loss: 0.0744
# train loss: 0.0717
def __model_12(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(8, activation='relu'))

    # decoder
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', kernel_initializer='he_uniform', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    model.save("final_trained_model")

    return model, training_error


# Model 13 
# added some regularizers -> still works good but the previous one without regularization was better
# val loss: 0.0788
# train loss: 0.0771
def __model_13(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2=0.001)))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


# Model 14 
# added some regularizers -> still works good but the previous one without regularization was better
# val loss: 0.0758
# train loss: 0.0730
def __model_14(data):
    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data
    model = Sequential()

    # encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(49, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(4, activation='relu'))

    # decoder
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Reshape((7, 7, 1)))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=100, batch_size=512, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


def __model_15(data: Tuple[np.ndarray]):
    """ Different approach base model

        Training loss: 0.1574 - val_loss: 0.1575
    """
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


def __model_16(data: Tuple[np.ndarray]):
    """ Reduced batch_size for better convergence. Latent space dimension was initially very low (doubled to 8).

        Training loss: 0.14 Validation loss: 0.14 (6 epochs)
    """

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    # create model
    weight_regularizer = L1L2(0.0001)
    latent_space_dim = 8

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
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=25, batch_size=32, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


def __model_17(data: Tuple[np.ndarray]):
    """ Doubled latent space dimension again (8 -> 16).

        Training loss: 0.16 Validation loss: 0.16 (12 epochs)
    """

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    # create model
    weight_regularizer = L1L2(0.0001)
    latent_space_dim = 8

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
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=25, batch_size=32, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


def __model_18(data: Tuple[np.ndarray]):
    """ Doubled latent space dimension again (16 -> 32).

        Training loss: 0.16 Validation loss: 0.16 (16 epochs)
    """

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    # create model
    weight_regularizer = L1L2(0.0001)
    latent_space_dim = 8

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
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=25, batch_size=32, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


def __model_19(data: Tuple[np.ndarray]):
    """ latent_space = 8; batch_size = 32; replaced activity regularizer & stochastic dropout with batch normalization

        Training loss: 0.09 Validation loss: 0.14 (7 epochs)
    """

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = data

    # create model
    weight_regularizer = L1L2(0.0001)
    latent_space_dim = 8

    model = Sequential()

    # encoder
    model.add(Conv2D(filters=32,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                     kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                     # 3) instead of once (5, 5) is a modern efficient approach)
                     strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                     padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                     activation='relu',  # relu is considered a standard for conv layers and typically works really well
                     kernel_initializer='he_uniform',  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
                     # 2)/num_inputs
                     input_shape=(28, 28, 1)))  # needs to be defined since it's the first layer of the network
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.95, epsilon=0.001))
    model.add(Conv2D(filters=32,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                     kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                     # 3) instead of once (5, 5) is a modern efficient approach)
                     strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                     padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                     activation='relu',  # relu is considered a standard for conv layers and typically works really well
                     kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.95, epsilon=0.001))
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
                              kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(BatchNormalization(momentum=0.95, epsilon=0.001))
    model.add(UpSampling2D((2, 2)))  # inverse operation of MaxPooling2D in encoder
    model.add(Conv2DTranspose(filters=1,  # number of convolutional kernels/channels; filters = {8, 16, 32, 64}
                              kernel_size=(3, 3),  # typical values in modern architectures: (1, 1), (3, 3), (5, 5), (7, 7) ... (3, 3) with stacked convolutional layers (e.g. twice applying (3,
                              # 3) instead of once (5, 5) is a modern efficient approach)
                              strides=(1, 1),  # using the smallest convolutional stride since our input images are of small dimensions
                              padding='same',  # using 'same' instead of default 'valid' padding automatically introduces the needed padding so that the full pixel information is used
                              activation='relu',  # relu is considered a standard for conv layers and typically works really well
                              kernel_initializer='he_uniform'))  # good practice for convolution: initialization of kernel with a uniform distribution centered on 0 with std_deviation = sqrt(
    # 2)/num_inputs
    model.add(BatchNormalization(momentum=0.95, epsilon=0.001))

    # Configure the model training procedure
    model.compile(loss=tf.keras.losses.MSE, optimizer='adam', metrics=[])
    model.summary()

    # Train and evaluate the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    training_error = model.fit(x_train_perturb, x_train_transformed, epochs=25, batch_size=32, validation_data=(x_val_perturb, x_val_transformed), callbacks=[es])

    return model, training_error


if __name__ == '__main__':
    perturbed_data_set = __create_perturbed_data_set()
    __train_autoencoder(perturbed_data_set)
    # __create_comparison_images(perturbed_data_set, 1000)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GaussianNoise, BatchNormalization, Reshape, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
import os

RANDOM_STATE = 42
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      #pls dont delete this


def __train_autoencoder(perturbed_data_set):

    x_train_transformed, x_val_transformed, x_test_transformed, x_train_perturb, x_val_perturb, x_test_perturb = perturbed_data_set

    model, training_error = __base_model(x_train_transformed)
    testing_error = model.evaluate(x_val_transformed, x_val_transformed, batch_size=64)

    # Print the final training error, validation error and test accuracy
    print('Training error: '"{:.2f}".format(training_error.history['loss'][-1]) + " Validation error: " + "{:.2f}".format(training_error.history['val_loss'][-1]))
    print('Test accuracy: ' + "{:.4f}".format(testing_error[1]))

    # Plot the evolution of the error during training
    plt.plot(training_error.history['loss'])
    plt.plot(training_error.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss', loc='top')
    plt.xlabel('epoch', loc='right')
    plt.legend(['Training error', 'Validation error'], loc='upper right')
    plt.annotate('Test error: ' + "{:.2f}".format(testing_error[0]) + ' Test accuracy: ' + "{:.4f}".format(testing_error[1]), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points',
                 va='top', color='red')
    plt.show()


def __create_perturbed_data_set():
    np.random.seed(RANDOM_STATE)

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

    noise = np.random.normal(0.0, stddev, size=28*28).reshape(image.shape)
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


def __base_model(x_train_transformed):
    # Configure the model
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
    training_error = model.fit(x_train_transformed, x_train_transformed, epochs=100, batch_size=64, callbacks=[es])
    return model, training_error


if __name__ == '__main__':
    perturbed_data_set = __create_perturbed_data_set()
    __train_autoencoder(perturbed_data_set)

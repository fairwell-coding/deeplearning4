import pickle

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GaussianNoise, BatchNormalization, Reshape, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def __train_autoencoder():
    # Load data
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

    # TODO
    # 1. Gaussian noise Sebastian
    # 2. occlusion of image part: black square Philipp
    # 3. brightness Sebastian
    # 4. rotation Clemens
    # 5. horizontal/vertical flip Clemens

    data = None
    with open('./perturbed_fashion_mnist', mode='wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    # __train_autoencoder()
    __create_perturbed_data_set()

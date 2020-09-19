"""
A Convolutional Neural Network class that recognizes handwritten letters. 
"""

import numpy as np

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation, MaxPool2D, AvgPool2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.regularizers import l2, l1
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
import os
import shutil
import cv2
import random
import pickle
import math
import gc
import tensorflow as tf
import tensorflow_datasets as tfds
from extra_keras_datasets import emnist
from PIL import Image

from scipy.io import loadmat
from keras.utils import np_utils
import idx2numpy

tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class CNN():
    def __init__(self):
        self.train_path = "/mnt/e/Images for CNN/train"
        self.test_path = "/mnt/e/Images for CNN/test"

    def set_training_data(self):
        img_size = 50
        test_data = []
        test_X = []
        test_Y = []
        for category in self.lables:
            path = os.path.join(self.test_path, category)

            for img in os.listdir(path):

                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    test_data.append([new_array, class_num])
                except Exception as e:
                    pass
        random.shuffle(test_data)
        for features, label in test_data:
            test_X.append(features)
            test_Y.append(label)
        test_X = np.array(test_X).reshape(-1, img_size, img_size, 1)
        np.save("testX.npy", test_X)
        np.save("testY.npy", test_Y)

    def reset(self):
        sess = K.get_session()
        k.clear_session()
        sess.close()
        sess = get_session()
        try:
            del model
        except:
            pass
        print(gc.collect())
        tf.config.optimizer.set_jit(True)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    def train(self):
        batch = 64
        train_datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.10, width_shift_range=0.10, rescale=1./255, height_shift_range=0.10)
        test_datagen = ImageDataGenerator(rescale=1./255)

        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1),
                         activation="relu", strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))


        model.add(Conv2D(64, (5, 5), activation="relu",
                         strides=2, padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (4, 4), activation="relu"))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dropout(0.4))

        model.add(Dense(37, activation='softmax'))

        model.summary() 
        lrate = 0.01
        adam = keras.optimizers.SGD(lr=lrate)
        epochs = 100
        annealer = ReduceLROnPlateau(
            monitor='val_loss', mode='min',  factor=0.95, patience=3, verbose=1)
        model.compile(loss="categorical_crossentropy",
                      optimizer=adam, metrics=['accuracy'])
        csv_logger = CSVLogger('training.log')
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        mc = ModelCheckpoint('model.h5', monitor='val_loss',
                             mode='min', save_best_only=True)

        model.fit(train_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/train', target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=True), epochs=epochs,
                  validation_data=test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/test', target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=False), callbacks=[es, mc, annealer, csv_logger])
        for i in range(1, 5):
            if (i % 2 == 0):
                lrate = lrate / 10

            K.clear_session()

            del model
            tf.config.optimizer.set_jit(True)

            gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

            model = load_model('model.h5')

            adam = keras.optimizers.SGD(lr=lrate, momentum=0.95, nesterov=True)
            model.compile(loss="categorical_crossentropy",
                          optimizer=adam, metrics=['accuracy'])

            model.fit(train_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/train', target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=True), epochs=epochs, validation_data=test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/test', target_size=(28, 28), batch_size=batch, color_mode='grayscale', class_mode='categorical', shuffle=False),
                      callbacks=[es, mc, annealer,  csv_logger])

        model.save("model_test.h5")

    def test(self):
        from keras.preprocessing import image
        model = load_model('model.h5', compile=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/training', target_size=(28, 28), batch_size=32,
                                                           color_mode='grayscale', class_mode='categorical', shuffle=False)

        test_loss, test_acc = model.evaluate_generator(
            train_generator, verbose=1)
        print('\nTest Loss: ', test_loss)
        print('\nTest accuracy: ', test_acc)

    def predict(self):
        IMG_SIZE = 28  # 50 in txt-based
        img_array = Image.open('test-J.jpg').convert("L")
        new_array = np.reshape(img_array, (28, 28, 1))
        im2arr = np.array(new_array)
        im2arr = im2arr.reshape(-1, 28, 28, 1)
        model = load_model('model.h5')
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = test_datagen.flow_from_directory('Images for CNN/imageTest/Output/balanced/training', target_size=(28, 28), batch_size=32,
                                                           color_mode='grayscale', class_mode='categorical', shuffle=False)
        y_Labels = train_generator.class_indices
        y_Labels = {y: x for x, y in y_Labels.items()}
        print(type(y_Labels))
        prediction = model.predict_classes([im2arr])
        print(type(prediction))
        return y_Labels[prediction[0]]


if __name__ == "__main__":

    temp = CNN()
    # temp.train()
    # temp.guess()

    print("This is prediction")
    print(temp.prepare())
    print("------")
    print("Done!!!")

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time

NAME = 'cat-vs-dog-cnn-64x2-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

dir = '/Users/jasonchang/Desktop/PycharmProjects/deep-learning/data/PetImages'
labels = ['Dog', 'Cat']

training_data = []


def create_training_dataset():
    global x, y

    for label in labels:
        # joined path to cats or dogs folder
        path = os.path.join(dir, label)
        class_num = labels.index(label)

        for image in os.listdir(path):
            try:
                # input the image in grayscale to reduce space complexity
                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

                # print(img_array)
                # print(img_array.shape)
                # plt.imshow(img_array, cmap='gray')
                # plt.show()

                IMG_SIZE = 50

                # standardize the dimension of each image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # plt.imshow(new_array, cmap='gray')
                # plt.show()

                training_data.append([new_array, class_num])

            except Exception as e:
                # pass the broken images for now
                pass

    random.shuffle(training_data)

    print(len(training_data))

    for sample in training_data[:10]:
        print(sample[1])

    x = []
    y = []

    for features, classes in training_data:
        x.append(features)
        y.append(classes)

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open('x.pickle', 'wb')
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open('y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()


try:
    pickle_in = open('x.pickle', 'rb')
    x = pickle.load(pickle_in)
    print(x[1])
    print(x[1].shape)

    pickle_in = open('y.pickle', 'rb')
    y = pickle.load(pickle_in)
    print(y)
    print(np.array(y).shape)

except IOError as e:
    create_training_dataset()


# range of pixel value -> [0, 255.0]
x = x / 255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Converting 3D feature maps to 1D feature vectors
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
Validation Accuracy -> Out-sample forecasting
Training Accuracy -> In-sample forecasting
'''
model.fit(x, y, batch_size=64, epochs=10,
          validation_split=0.3, callbacks=[tensorboard])












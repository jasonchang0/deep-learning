import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

dir = '/Users/jasonchang/Desktop/PycharmProjects/deep-learning/data/PetImages'
labels = ['Dog', 'Cat']

training_data = []


def create_training_dataset():
    for label in labels:
        # joined path to cats or dogs folder
        path = os.path.join(dir, label)
        class_num = labels.index(label)

        for image in os.listdir(path):
            try:
                # input the image in grayscale to reduce space complexity
                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

                print(img_array)
                print(img_array.shape)
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

            for features, label in training_data:
                x.append(features)
                y.append(label)

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
    print(y[1])
    print(y[1].shape)

except IOError as e:
    create_training_dataset()









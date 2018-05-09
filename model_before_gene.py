import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import utils
from PIL import Image


def augment_data(images, measurements):
    augmented_images = [] + images
    augmented_measurements = [] + measurements
    for image, steering_angle in zip(images, measurements):
        flipped_image, flipped_steering_angle = flip_image_steering(image, steering_angle)
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_steering_angle)
    utils.beep()
    return augmented_images, augmented_measurements


def flip_image_steering(image, steering_angle):
    flipped_image = np.fliplr(image)
    flipped_steering_angle = steering_angle * -1.0
    return flipped_image, flipped_steering_angle


def load_csv_data(file_path):
    # Reading the recorded data from the .csv file
    lines = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def get_images_and_measurements(lines):
    images = []
    measurements = []

    for line in lines:
        steering_center_angel = float(line[3])

        # Adjusted Steering angels for side camera images
        correction_factor = 0.2  # should change this to a computed parameter
        steering_left_angel = steering_center_angel + correction_factor
        steering_right_angel = steering_center_angel - correction_factor

        # Read in the images form center, left, and right cameras
        center_image = np.asarray(Image.open(line[0]))
        left_image = np.asarray(Image.open(line[1]))
        right_image = np.asarray(Image.open(line[2]))
        # np.asarray(Image.open(path + row[0]))
        # image = cv2.imread(source_path)

        # Add images and angels to the dataset
        images.extend([center_image, left_image, right_image])
        measurements.extend([steering_center_angel, steering_left_angel, steering_right_angel])
    return images, measurements

#  Keras Data generator

#
# def data_generator(samples, batch_size):
#     n_samples = len(samples)
#     while 1: # Forever loop to keep the generator up till the termination of the program
#              #(end of training and inference)
#         # Shuffle the data before bedfore batching batch data
#         shuffle(samples)
#         for offset in range(0, n_samples, batch_size):
#             # Create batch of batch_size
#             batch_samples = samples[offset : offset + batch_size]
#             # Get images and measurements (angels) for the batch
#             batch_images, batch_measurements = get_images_and_measurements(batch_samples)
#             # Augment the batch dataset
#             augmented_batch_images, augmented_batch_measurements = augment_data(images, measurements)
#             # Putting our augmented data into numpy arrays cause Keras require numpy arrays
#             batch_features = np.array(augmented_batch_images)
#             batch_labels = np.array(augmented_batch_measurements)
#             # Shuffle the batch data for good measure
#             yield shuffle(batch_features, batch_labels)

    # Load data from csv file
lines = load_csv_data('./DrivingData/driving_log.csv')
# Getting the images and measurements
images, measurements = get_images_and_measurements(lines)
# Augment the data using fliplr
augmented_images, augmented_measurements = augment_data(images, measurements)
print('We used to have {} images and angels'.format(len(images)))
print('Now we have {} augmented images, and {} augmented angels'.format(len(augmented_images),
                                                                        len(augmented_measurements)))
# Putting our augmented data into numpy arrays cause Keras require numpy arrays
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#
# # Shuffling the data
X_train, y_train = shuffle(X_train, y_train, random_state=0)

# img, steer = flip_image_steering(X_train[0], y_train[0])
# print(y_train[0])
# print(steer)
# plt.imshow(cv2.cvtColor(X_train[0], cv2.COLOR_BGR2RGB))
# # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
# print(y_train[9])
# Keras LeNet Model

model = Sequential()
# Normalizing and standardizing our images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Cropping our images using Cropping2D
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# First Convolution2D layer with
model.add(Convolution2D(6, 5, 5, activation='relu'))
# MaxPooling2D layer
model.add(MaxPooling2D())
# Second Convolution2D layer with
model.add(Convolution2D(6, 5, 5, activation='relu'))
# MaxPooling2D layer
model.add(MaxPooling2D())
# Flattening the Images after the convolutional steps
model.add(Flatten())
# Fist dense layer
model.add(Dense(120))
# Second dense layer
model.add(Dense(84))
# Logits layer
model.add(Dense(1))

# Hyperparameters
EPOCHS = 20
BATCHSIZE = 256

model.compile(loss='mse', optimizer='adam')
model_history = model.fit(X_train,
                          y_train,
                          validation_split=0.2,
                          shuffle=True,
                          batch_size=BATCHSIZE,
                          nb_epoch=EPOCHS)
print(model_history.history.keys())
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model2.h5')

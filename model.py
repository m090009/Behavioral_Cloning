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
import math


def augment_data(images, measurements):
    augmented_images = []  # + images
    augmented_measurements = []  # + measurements
    for image, steering_angle in zip(images, measurements):
        flipped_image, flipped_steering_angle = flip_image_steering(image, steering_angle)
        augmented_images.append(flipped_image)
        augmented_images.append(image)
        augmented_measurements.append(flipped_steering_angle)
        augmented_measurements.append(steering_angle)
    # utils.beep()
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


def data_generator(samples, batch_size):
    n_samples = len(samples) * 3 * 2  # number of samples * 3 camera images * augmentation
    # print('Number of samples')
    while 1:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and inference)
        # Shuffle the data before bedfore batching batch data
        # if n_samples == 29880:
        #     print('\nTraining\n===========================')
        # else:
        #     print('\nValidation\n=========================')
        # print('Number of samples {}'.format(n_samples))
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            # Create batch of batch_size
            batch_samples = samples[offset: offset + batch_size]
            # Get images and measurements (angels) for the batch
            batch_images, batch_measurements = get_images_and_measurements(batch_samples)
            # Augment the batch dataset
            augmented_batch_images, augmented_batch_measurements = augment_data(batch_images,
                                                                                batch_measurements)
            # Putting our augmented data into numpy arrays cause Keras require numpy arrays
            batch_features = np.array(augmented_batch_images)
            batch_labels = np.array(augmented_batch_measurements)
            # Shuffle the batch data for good measure
            print(' X_train: {} and y_train: {}'.format(batch_features.shape, batch_labels.shape))
            yield shuffle(batch_features, batch_labels)


def data_generator1(samples, batch_size):
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and inference)
        shuffle(samples)
        X_data = []
        y_data = []
        for i, sample in enumerate(samples):
            # Get the samples images, which will return 3 images (center, left, right)
            # and their angles
            sample_images, sample_measurements = get_images_and_measurements([sample])
            # Augment sample images (flip)
            augmented_sample_images, augmented_sample_measurements = augment_data(sample_images,
                                                                                  sample_measurements)
            # Adding our generated sample data into our yield arrays
            X_data.extend(augmented_sample_images)
            y_data.extend(augmented_sample_measurements)
            # print('X_data length: {}'.format(len(X_data)))
            # Check if X is of batch_size or if its the last element
            if len(X_data) > batch_size or i == len(samples) - 1:
                # print('==================Batch====================')
                # Putting our augmented data into numpy arrays cause Keras require numpy arrays
                # yield the batch
                # Shuffle the batch data for good measure
                yield shuffle(np.array(X_data[:batch_size]), np.array(y_data[:batch_size]))
                X_data = X_data[batch_size:]
                y_data = y_data[batch_size:]


def plot_loss(model_history):
    print(model_history.history.keys())
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# Hyperparameters
EPOCHS = 20
BATCHSIZE = 512

# Load data from csv file
lines = load_csv_data('./DrivingData/driving_log.csv')
# Splitting the data to a 80% training and 20% validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
# print('train_samples {}, validation_samples {}'.format(len(train_samples), len(validation_samples)))
train_generator = data_generator1(train_samples, batch_size=BATCHSIZE)
validation_generator = data_generator1(validation_samples, batch_size=BATCHSIZE)

# Keras LeNet Model
model = Sequential()
# Normalizing and standardizing our images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Cropping our images using Cropping2D
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# First Convolution2D layer with
model.add(Convolution2D(6, (5, 5), activation='relu'))
# MaxPooling2D layer
model.add(MaxPooling2D())
# Second Convolution2D layer with
model.add(Convolution2D(6, (5, 5), activation='relu'))
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
# Defining the loss function and optimizer
model.compile(loss='mse', optimizer='adam')

training_lenght = math.ceil((len(train_samples)*3*2) / BATCHSIZE)
validation_length = math.ceil((len(validation_samples)*3*2) / BATCHSIZE)
model_history = model.fit_generator(train_generator,
                                    steps_per_epoch=training_lenght,
                                    validation_data=validation_generator,
                                    validation_steps=validation_length,
                                    epochs=EPOCHS, verbose=1)

model.save('model.h5')
plot_loss(model_history=model_history)

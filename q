def data_generator(samples, batch_size):
    # print('Number of samples')
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and inference)
        shuffle(samples)
        images, measurements = get_images_and_measurements(samples)
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


def data_generator(samples, batch_size):
    # print('Number of samples')
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and inference)
        shuffle(samples)
        X_data = []
        y_data = []
        for i, sample in enumerate(samples)
        shuffle(samples)
        # Get the samples images, which will return 3 images (center, left, right)
        # and their angles
            sample_images, sample_measurements = get_images_and_measurements([sample])
            # Augment sample images (flip)
            augmented_sample_images, augmented_sample_measurements = augment_data(sample_images,
                                                                                  sample_measurements)

            # Adding our generated sample data into our yield arrays
            X_data.extend(augmented_sample_images)
            y_data.extend(augmented_sample_measurements)

            # Check if X is of batch_size or if its the last element
            if len(X_data) > batch_size or i == len(samples) - 1:
                # Putting our augmented data into numpy arrays cause Keras require numpy arrays
                batch_features = np.array(X_data)
                batch_labels = np.array(y_data)
                # yield the batch
                # Shuffle the batch data for good measure
                print('Batch size {}'.format())
                yield shuffle(batch_features, batch_labels)
                X_data = []
                y_data = []


def another_generator(samples, batch_size):
    num_samples = len(samples) * 3 * 2
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        samples_counter = 0
        for offset in range(0, num_samples, batch_size):
            # batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

import csv
import cv2
import numpy as np

samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            measurements = []
            for sample in batch_samples:
                filename = sample[0].split('/')[-1]
                current_path = './data/IMG/' + filename
                center_img = cv2.imread(current_path)
                center_angle = float(sample[3])
                images.append(center_img)
                measurements.append(center_angle)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)
#print("data ready")

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print("layer ready")

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    epochs=3)

model.save('model.h5')

#predict = model.predict(X_train, batch_size=128)
#print(predict)
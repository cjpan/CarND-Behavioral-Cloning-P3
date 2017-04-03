import pandas as pd
import cv2
import numpy as np

log_name = './data/driving_log.csv'
log_data = pd.read_csv(log_name)

def random_selection(x):
    return x[3] != 0.0 or np.random.randint(0,100) < 10

samples = list(filter(random_selection, np.array(log_data.values)))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from sklearn.utils import shuffle

cameras = ['center', 'left', 'right']
steer_offset = [0, -.25, .25]

def generator(samples, batch_size=32, augment=True):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            steers = []

            for sample in batch_samples:
                camera = np.random.choice(cameras)
                filename = sample[camera].split('/')[-1]
                current_path = './data/IMG/' + filename
                img = cv2.imread(current_path)[65:135]
                img = cv2.resize(img, (64,64))
                steer = float(sample[3] + steer_offset[camera])
                images.append(img)
                steers.append(steer)

            all_images, all_steers = [], []
            for image, steer in zip(images, steers):
                all_images.append(image)
                all_steers.append(steer)

                if augment:
                    flip_img = cv2.flip(image, 1)
                    flip_steer = steer * -1.0
                    all_images.append(flip_img)
                    all_steers.append(flip_steer)

                    #shadow = image * 0.5
                    #all_images.append(shadow)
                    #all_steers.append(steer)

                    #flip_shadow = flip_img * 0.5
                    #all_images.append(flip_shadow)
                    #all_steers.append(flip_steer)

            X_train = np.array(all_images)
            y_train = np.array(all_steers)
            yield shuffle(X_train, y_train)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size, augment=False)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(.25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam())
print(len(train_samples), len(validation_samples))
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size,
                    epochs=1)

model.save('model.h5')
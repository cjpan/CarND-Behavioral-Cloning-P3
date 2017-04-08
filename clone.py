import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

log_name = './data/driving_log.csv'
log_data = pd.read_csv(log_name)
samples = np.array(log_data.values)


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# 20% of data for validation.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def random_selection(x):
    # for data with tiny steering angles, randomly select 20%
    return abs(x[3]) > 0.1 or np.random.randint(0,100) < 20

train_samples = list(filter(random_selection, np.array(train_samples)))

# offsets for different cameras
cameras = [0, 1, 2]
steer_offset = [0, .25, -.25]


def train_generator(samples, batch_size=32):
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
                image = cv2.imread(current_path)
                image = image[65:135,:,:]
                # add offset for different cameras.
                steer = float(sample[3] + steer_offset[camera])
                if steer > 1:
                    steer = 1.
                if steer < -1:
                    steer = -1.

                # flip half of the images
                if np.random.randint(0,10) < 5:
                    image = cv2.flip(image, 1)
                    steer = -steer

                # randomly shift the image and adjust the steering angle.
                # 0.002 for per pixel.
                max_shift = 25
                max_steer = 0.05
                f = np.random.random_integers(-max_shift, max_shift)
                M = np.float32([[1,0,f],[0,1,0]])
                shift_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                steer += f / max_shift * max_steer
                image = shift_img
                if steer > 1:
                    steer = 1.
                if steer < -1:
                    steer = -1.

                # randomly adjust the brightness for each training image.
                bright_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                random_bright = np.random.uniform(.5, 1.5)
                bright_img[:,:,2] = bright_img[:,:,2] * random_bright
                bright_img = cv2.cvtColor(bright_img, cv2.COLOR_HSV2BGR)
                image = bright_img

                # add random shadows for each training image.
                h, w = image.shape[0], image.shape[1]
                v1 = [0, 0]
                v2 = [np.random.randint(0,w), 0]
                v3 = [np.random.randint(0,w), h]
                v4 = [0, 64]
                vertices = np.array([v1, v2, v3, v4])
                mask = cv2.fillConvexPoly(image, vertices, (0,0,0))
                alpha = np.random.uniform(0.25, 0.75)
                shadow_img = cv2.addWeighted(mask, alpha, image, 1-alpha, 0)
                image = shadow_img

                # resize training image to 66x200
                # and color space to RGB to fit the input of training network
                image = cv2.resize(image, (200,66))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images.append(image)
                steers.append(steer)

            X_train=np.array(images)
            y_train=np.array(steers)
            yield shuffle(X_train, y_train)

def validation_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            steers = []
            # use original images of center camera for validation.
            # Only size and color space is converted to fit the model.
            for sample in batch_samples:
                filename = sample[0].split('/')[-1]
                current_path = './data/IMG/' + filename
                image = cv2.imread(current_path)
                image = image[65:135,:,:]
                image = cv2.resize(image, (200,66))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                steer = float(sample[3])
                images.append(image)
                steers.append(steer)

            X_valid = np.array(images)
            y_valid = np.array(steers)
            yield shuffle(X_valid, y_valid)

batch_size = 128
train_gen = train_generator(train_samples, batch_size=batch_size)
validation_gen = validation_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(66,200,3)))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(.4))
model.add(Dense(10))
model.add(Dense(1))

print(model.summary())


model.compile(loss='mse', optimizer=Adam())
print(len(train_samples), len(validation_samples))
history_object = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_gen,
                    validation_steps=len(validation_samples)/batch_size,
                    epochs=10,
                    verbose=2)

model.save('model.h5')

# visualize the loss/val_loss.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
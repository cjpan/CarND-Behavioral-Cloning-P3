import pandas as pd
import cv2
import numpy as np

log_name = './data/driving_log.csv'
log_data = pd.read_csv(log_name)

samples = np.array(log_data.values)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)
from sklearn.utils import shuffle

def random_selection(x):
    return abs(x[3]) > 0.1 or np.random.randint(0,100) < 15

train_samples = list(filter(random_selection, np.array(train_samples)))

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
                #image = cv2.resize(image, (64,64))
                steer = float(sample[3] + steer_offset[camera])
                if steer > 1:
                    steer = 1.
                if steer < -1:
                    steer = -1.
                #images.append(image)
                #steers.append(steer)

                if np.random.randint(0,10) < 5:
                    image = cv2.flip(image, 1)
                    steer = -steer

                max_shift = 50
                max_steer = 0.1
                f = np.random.random_integers(-max_shift, max_shift)
                M = np.float32([[1,0,f],[0,1,0]])
                shift_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                steer += f / max_shift * max_steer
                image = shift_img
                if steer > 1:
                    steer = 1.
                if steer < -1:
                    steer = -1.
                #images.append(shift_img)
                #steers.append(shift_steer)

                bright_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                random_bright = np.random.uniform(.5, 1.5)
                bright_img[:,:,2] = bright_img[:,:,2] * random_bright
                bright_img = cv2.cvtColor(bright_img, cv2.COLOR_HSV2BGR)
                    #images.append(bright_img)
                    #steers.append(s)
                image = bright_img

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

batch_size = 64
train_gen = train_generator(train_samples, batch_size=batch_size)
validation_gen = validation_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

model = Sequential()
#model.add(Cropping2D(((,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(66,200,3)))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
#model.add(Dropout(.4))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(Dropout(.3))
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(.25))
model.add(Dense(10))
model.add(Dense(1))

# model = Sequential()
# # Normalize
# #model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64,64,3)))
# #model.add(Convolution2D(3,1,1,border_mode='valid', name='conv0'))
# # layer 1 output shape is 32x32x32
# model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), strides=(2, 2), padding='same', activation='relu'))
# # layer 2 output shape is 15x15x16
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
# model.add(Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
# # layer 3 output shape is 13x13x16
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
# model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
# # Flatten the output
# model.add(Flatten())
# # layer 4
# model.add(Dropout(.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(.5))
# # layer 5
# model.add(Dense(64, activation='relu'))
# # Finally a single output, since this is a regression problem
# model.add(Dense(1))

print(model.summary())


model.compile(loss='mse', optimizer=Adam())
print(len(train_samples), len(validation_samples))
model.fit_generator(train_gen,
                    steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_gen,
                    validation_steps=len(validation_samples)/batch_size,
                    epochs=10,
                    verbose=2)

model.save('model.h5')
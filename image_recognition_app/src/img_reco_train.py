#!/usr/bin/env python3

from keras.datasets import mnist
import matplotlib. pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')


# Load dataset (download if needed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# # To plot uncomment this
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# plt.show()

# fix the seed 
seed = 7
numpy.random.seed(seed)

# X_train.shape[0] means the number of images. 
# 1 means that the images are gray-scaled, not colored. If it would be colored than it should be 3.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalize the set of pixels
X_train = X_train / 255
X_test = X_test / 255

# one hot encoding
# output - [ 0 0 0 0 0 1 0 0 0 0 ] This corresponds to number 5.
# because the y_train and y_test have all categorical values from 0 to 9.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(8, (3,3), input_shape=(1,28,28), activation='relu')) # 8 Conv filter layers with the shape of 3x3.
    model.add(MaxPooling2D(pool_size=(2,2))) # reducing the shape half with the maxpooling.
    # pxqxr
    # Flatten takes for exp a 3x3 matrix and flattens to 9x1 matrix.
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) # we get each class's probabilities 
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model

# build a model
model = baseline_model()

# Fit 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3,
          batch_size=32, verbose=2)

model.save('./image_recognition_app/trained_model/model.h5')

# Final eval
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN error: %.2f%%" % (100 - scores[1]*100))















































































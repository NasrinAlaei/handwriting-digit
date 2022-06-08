# -*- coding: utf-8 -*-


!pip3 uninstall tensorflow
!pip3 install tensorflow.gpu==2.0.0

# Commented out IPython magic to ensure Python compatibility.

"""step is to import the classes and functions needed"""

from hoda_dataset_helper import read_hoda
from hoda_dataset_helper import __read_hoda_dataset
from hoda_dataset_helper import __read_hoda_cdb

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""we need to load and read the hoda dataset so that it is suitable for use training a CNN"""

# reading the hoda datasets:
# train dataset
X_train, Y_train = __read_hoda_dataset(dataset_path='Train 60000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=False ,
                                reshape=False)
# test dataset
X_test, Y_test = __read_hoda_dataset(dataset_path='Test 20000.cdb',
                              images_height=32,
                              images_width=32,
                              one_hot=False,
                              reshape=False)
# remain dataset
X_remain, Y_remain = __read_hoda_dataset('RemainingSamples.cdb',
                                             images_height=32,
                                             images_width=32,
                                             one_hot=False,
                                             reshape=False)

"""I have two simple one-dimensional arrays in NumPy. I should be able to concatenate them using numpy.concatenate"""

# concatenate
x_train=np.concatenate([X_train, X_remain])
y_train=np.concatenate([Y_train,Y_remain])

"""it is a good idea to normalize the pixel values to the range 0 and 1 and one hot encode the output variables."""

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255.0
X_test = X_test /255.0

"""Convert class vectors to binary class matrices:
the output variable is an integer from 0 to 9. this is a class.such this is good practice to use a one hot encoding of the class values, transforming the vector of class integers into a binary matrix.
"""

# one hot encode output
y_train = to_categorical(y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)

"""Split train datasset to train and validation"""

# split
x_train,x_val,y_train,y_val= train_test_split(x_train,y_train,test_size=0.1, random_state=2)

train_images, train_labels = __read_hoda_cdb('Train 60000.cdb')

test_images, test_labels = __read_hoda_cdb('Test 20000.cdb')

remaining_images, remaining_labels = __read_hoda_cdb('RemainingSamples.cdb')

train_img =np.concatenate([train_images,remaining_images])
label_img =np.concatenate([train_labels,remaining_labels])

"""I used the keras Sequential API

the first is the convolutional(Conv2D) layer. choosed to set 64 filters (5,5) for the two firsts conv2D layers and 32 filters (3,3) for the two last ones.  
the second important layer in CNN is the Maxpooling (2,2) in the each layers.
dropouts randomly a propotion of the network and forces the network to learn featurees in a distributed way.(dropout(0.25))

used the  one fullyconnected layer(dense) with 512 neuron and relu activation.
and finally used the output layer (is fullyconnected) with 10 neuron(class) and softmax activation .

"""


# build the CNN model
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same', activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(filters=64,kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32 , kernel_size=(3,3), padding='same', activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(filters=32 , kernel_size=(3,3), padding= 'same', activation='relu'))
model.add(MaxPool2D(pool_size =(2,2) , strides=(2,2)))
model.add(Dropout(0.25))

# classifier
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# define the optimize
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping=tf.keras.callbacks.EarlyStopping(patience=3, monitor="val_accuracy")

# set a learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                        patience=2, 
                        verbose=1, 
                        factor=0.4 ,
                        min_lr=0.00001)

"""without data augmentation i obtained an accuracy of

with data augmentation i achieved 99.11% of accuracy
"""

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=10,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(X_train)

model.summary()

tensorbord = tf.keras.callbacks.TensorBoard(log_dir='./logs')

my_callbacks=[learning_rate_reduction,tensorbord, early_stopping]

result=model.fit(x_train, y_train,epochs=30, batch_size=32, validation_data=(x_val,y_val), verbose=1, callbacks=my_callbacks)

# Evaluate and Predict Test data
model.evaluate(X_test, Y_test, verbose=1)

model.evaluate(x_val, y_val, verbose=1)

# Results
import pandas as pd
from sklearn.utils import shuffle

results = model.predict(X_test)
results = np.argmax(results,axis = 1)
results = pd.Series(results)

results = shuffle(results)

results

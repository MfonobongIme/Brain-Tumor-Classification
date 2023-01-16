#import libraries
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.utils import to_categorical

#image directory
image_directory = 'datasets/'

#create list for all images inside the no folder
no_tumor_images = os.listdir(image_directory + 'no/')

#create list for all images inside the yes folder
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = [] #dataset to store our images
label = [] #label to store our labels

INPUT_SIZE = 64

#print(no_tumor_images)

#path = 'no0.jpg'

#print(path.split('.')[1]) #prints the extension of the image file which is .jpg

# loop through the no folder and ensures its only the files with the jpg extension
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name) #read images
        image = Image.fromarray(image, 'RGB') #convert TO rgb 
        image = image.resize((INPUT_SIZE, INPUT_SIZE)) #resize image
        dataset.append(np.array(image)) #append the images in the dataset variable
        label.append(0) #appends zero which means no tumor


# loop through the yes folder and ensures its only the files with the jpg extension
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name) #read images
        image = Image.fromarray(image, 'RGB') #convert TO rgb 
        image = image.resize((INPUT_SIZE, INPUT_SIZE)) #resize image
        dataset.append(np.array(image)) #append the images in the dataset variable
        label.append(1) #appends one which means there is tumor

dataset = np.array(dataset) #convert to np array
label = np.array(label) #convert to np array


#divide model into train test split
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 0)

print(x_train.shape)

#normalize data for training
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# ------------------------------------------------------------Model Building ----------------------------------------------

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) #note use 2 dense layers if youre going to be using categorical cross entropy since we are dealing with binary problem
model.add(Activation('softmax'))


####Binary cross entropy = 1, sigmoid
####categorical cross entropy = 2, softmax

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 16, verbose = 1, epochs = 10, validation_data = (x_test, y_test), shuffle = False)

model.save('BrainTumor10EpochsCategorical.h5')



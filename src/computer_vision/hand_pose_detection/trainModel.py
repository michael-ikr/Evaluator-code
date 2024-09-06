import tensorflow as tf
import splitfolders as sp
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


import os

#sp.ratio('ImageSet', output = 'PracNewImageSet', seed=249, ratio = (0.7, 0.3), group_prefix=None, move=False)

train_data_dir = 'NewImageSet/train'
validation_data_dir = 'NewImageSet/val'

img_rows, img_cols = 28, 28
num_classes = 3
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=False,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode = 'binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size = batch_size,
    color_mode = 'grayscale',
    class_mode = 'binary'
)


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation= 'sigmoid'))

print(model.summary())

model.compile(loss = 'binary_crossentropy', 
              optimizer= 'rmsprop', 
              metrics= ['accuracy'])

nb_train_samples = 277
nb_validation_samples = 121
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples

)

model.save('draft_gestures.h5')


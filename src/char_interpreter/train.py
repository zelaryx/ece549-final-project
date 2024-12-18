import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, ReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from sklearn.utils import shuffle
import pickle

EMNIST_DIR = "/home/wshen2011/.cache/kagglehub/datasets/crawford/emnist/versions/3/"

train_samples = 1047366
num_epochs = 10
batch_size = 128
steps_per_epoch = train_samples // batch_size

label_map = pd.read_csv(EMNIST_DIR+'emnist-bymerge-mapping.txt', sep='\s+', header=None)
# create a dictionary that maps each index to its corresponding ASCII character. 
mapping_dict = {row[0]: chr(row[1]) for _, row in label_map.iterrows()}

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

def train_data_generator(file_path, batch_size=128, shuffle_data=True):
    while True:
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            if shuffle_data:
                chunk = shuffle(chunk)
            x_batch = np.array(chunk.iloc[:, 1:].values)

            x_batch = x_batch.reshape(-1, 28, 28, 1)
            y_batch = tf.keras.utils.to_categorical(np.array(chunk.iloc[:, 0].values), len(mapping_dict))

            datagen.fit(x_batch)
            augmented_data = datagen.flow(x_batch, y_batch, batch_size=batch_size)

            # Yield augmented batch
            x_augmented, y_augmented = next(augmented_data)

            yield x_augmented, y_augmented

train_generator = train_data_generator('letter_train_shuffle.csv', batch_size=batch_size, shuffle_data=True)

test_data = pd.read_csv('letter_test.csv')
x_test = np.array(test_data.iloc[:,1:].values)
y_test = np.array(test_data.iloc[:,0].values)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test, len(mapping_dict))

def create_sequential_cnn(input_shape, is_training=True):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Define CNN layer parameters
    kernel_sizes = [5, 5, 3, 3, 3]
    feature_maps = [32, 64, 128, 128, 256]
    pool_sizes = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 1)]
    strides = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 1)]

    # Add each layer to the Sequential model
    for i in range(len(kernel_sizes)):
        model.add(
            Conv2D(
                filters=feature_maps[i],
                kernel_size=(kernel_sizes[i], kernel_sizes[i]),
                strides=(1, 1),
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)
            )
        )
        
        model.add(BatchNormalization())
        
        # Wrap the batch norm layer to use `training=is_training`
        # at runtime if needed
        model.add(ReLU())
        
        if i == 4:
            model.add(
                MaxPooling2D(
                    pool_size=pool_sizes[i],
                    strides=strides[i],
                    padding='same'
                )
            )
        else:
            model.add(
                MaxPooling2D(
                    pool_size=pool_sizes[i],
                    strides=strides[i],
                    padding='valid'
                )
            )

    return model

model = create_sequential_cnn((28, 28, 1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(mapping_dict), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, validation_data=(x_test, y_test), epochs=num_epochs, callbacks=[early_stopping])

folder_name = f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}"
directory_name = os.path.join("models",folder_name)
os.makedirs(directory_name, exist_ok=True)

model.save(os.path.join(directory_name,'full_dataset.keras'))
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

with open(os.path.join(directory_name,'full_dataset_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist  # You can change to emnist for letters and digits
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import os
import natsort

def process_image(image_path, show=False):
    # image = load_img(image_path, target_size=(56, 56), color_mode="grayscale")
    # image = img_to_array(image).astype('float32')
    # image = np.pad(image, pad_width=((10, 10), (10, 10), (0, 0)), mode='constant', constant_values=255)
    image = load_img(image_path, target_size=(56, 56), color_mode="grayscale")
    image = img_to_array(image).astype('float32')
    image = np.pad(image, pad_width=((16, 16), (16, 16), (0, 0)), mode='constant', constant_values=255)

    image = tf.image.resize(image, [56, 56])

    image = image.numpy()

    kernel = np.ones((2, 2), np.float32)
    image = cv2.erode(image, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=2)

    image = 255.0 - image

    image = np.where(image > 255/3, image, 0)

    image = np.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [28, 28])

    image = image.numpy()

    if show:
        plt.imshow(image)
        plt.show()

    image = np.reshape(image, (1, 28, 28, 1))
    return image

def interpret_characters(directory):
    BASE_DIR = "/home/wshen2011/.cache/kagglehub/datasets/crawford/emnist/versions/3/"
    label_map = pd.read_csv(BASE_DIR+'emnist-bymerge-mapping.txt', sep='\s+', header=None)
    # create a dictionary that maps each index to its corresponding ASCII character. 
    alphabets_mapper = {row[0]: chr(row[1]) for _, row in label_map.iterrows()}

    # Load the model architecture and weights together
    model = tf.keras.models.load_model('models/full_dataset_20241206_0125/full_dataset.keras')
    # model = tf.keras.models.load_model('models/full_dataset_20241113_1436/full_dataset.keras')

    string_out = ""
    for word_dir in natsort.natsorted(os.listdir(directory)):
        for char_file in natsort.natsorted(os.listdir(os.path.join(directory,word_dir))):
            if char_file.lower().endswith('.jpg') or char_file.lower().endswith('.png'):
                img_path = os.path.join(directory, word_dir, char_file)
                image = process_image(img_path)

                # Run the model on the image
                predictions = model.predict(image)

                # Find the class with the highest probability
                # print(predictions)
                predicted_class = np.argmax(predictions)
                predicted_char = str(alphabets_mapper[predicted_class])
                string_out += predicted_char
                ending = ""
                if char_file.lower().endswith('.jpg'):
                    ending = ".jpg"
                else:
                    ending = ".png"
                new_file_name = f"{predicted_char}_{counter}{ending}"
                new_file_path = os.path.join(directory, word_dir, new_file_name)
                os.rename(img_path, new_file_path)
                counter += 1
        string_out += " "
    return string_out

print(interpret_characters("../ece549-final-project/output_lowercase"))
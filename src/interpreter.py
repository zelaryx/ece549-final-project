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

def process_image(image_path, show=False):
    image = load_img(image_path, target_size=(56, 56), color_mode="grayscale")
    image = img_to_array(image).astype('float32')
    image = np.pad(image, pad_width=((10, 10), (10, 10), (0, 0)), mode='constant', constant_values=255)
    # Rescale the padded image back to 28x28 pixels
    image = tf.image.resize(image, [56, 56])

    image = image.numpy()
    # # Normalize the pixel values
    # image = (255.0 - image*1.0)

    # Define a kernel for dilation
    kernel = np.ones((2, 2), np.float32)
        
    # Apply erosion
    image = cv2.erode(image, kernel, iterations=2)

    # Apply dilation to the eroded image
    image = cv2.dilate(image, kernel, iterations=2)

    image = 255.0 - image

    image = np.where(image > 255/4, image, 0)

    image = np.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [28, 28])

    # # Convert the rescaled image to a NumPy array
    image = image.numpy()

    if show:
        plt.imshow(image)
        plt.show()

    # Reshape the image to match the model's input shape (1, 28, 28, 1)
    image = np.reshape(image, (1, 28, 28, 1))
    return image

def interpret_characters(directory):
    BASE_DIR = "./char_interpreter/"
    label_map = pd.read_csv(BASE_DIR+'emnist-bymerge-mapping.txt', sep='\s+', header=None)
    # create a dictionary that maps each index to its corresponding ASCII character. 
    alphabets_mapper = {row[0]: chr(row[1]) for _, row in label_map.iterrows()}

    # Load the model architecture and weights together
    model = tf.keras.models.load_model(BASE_DIR+'full_dataset_20241113_1436/full_dataset.keras')

    string_out = ""
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = process_image(img_path)

            # Run the model on the image
            predictions = model.predict(image)

            # Find the class with the highest probability
            # print(predictions)
            predicted_class = np.argmax(predictions)

            string_out += str(alphabets_mapper[predicted_class])

    return string_out
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import subprocess

BASE_DIR = "/home/wshen2011/.cache/kagglehub/datasets/crawford/emnist/versions/3/"
label_map = pd.read_csv(BASE_DIR+'emnist-bymerge-mapping.txt', sep='\s+', header=None)
mapping_dict = {row[0]: chr(row[1]) for _, row in label_map.iterrows()}


#EMNIST DATASET
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.flatten()

for train_data in pd.read_csv(BASE_DIR+'emnist-bymerge-train.csv', chunksize=1000):
    train_data = shuffle(train_data)
    X = np.array(train_data.iloc[:, 1:].values)

    X = np.apply_along_axis(rotate, 1, X)
    Y = np.array(train_data.iloc[:, 0].values)

    with open("letter_train.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack((Y,X)))

for test_data in pd.read_csv(BASE_DIR+'emnist-bymerge-test.csv', chunksize=1000):
    X = np.array(test_data.iloc[:, 1:].values)

    X = np.apply_along_axis(rotate, 1, X)
    Y = np.array(test_data.iloc[:, 0].values)

    with open("letter_test.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack((Y,X)))

#KAGGLE CAPITAL LETTER DATASET
SECOND_DIR = "/home/wshen2011/.cache/kagglehub/datasets/sachinpatel21/"

def convert_cap_classification(y):
    return y + 10

for train_data_2 in pd.read_csv(SECOND_DIR+'az-handwritten-alphabets-in-csv-format/versions/5/A_Z Handwritten Data.csv', chunksize=1000):
    X = np.array(train_data_2.iloc[:,1:].values)
    # y_train = np.array(train_data.iloc[:,0].values)
    Y = np.array(train_data_2.iloc[:,0].values)
    train_data_2 = None
    Y = np.array([convert_cap_classification(y) for y in Y])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    with open("letter_train.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack((y_train,x_train)))
    
    with open("letter_test.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack((y_test,x_test)))

inverted_dict = {v: k for k, v in mapping_dict.items()}
print(inverted_dict)

def convert_lowercase_classification(y):
    if y in inverted_dict.keys():
        return inverted_dict[y]
    elif y.upper() in inverted_dict.keys():
        return inverted_dict[y.upper()]
    else:
        raise Exception


#STANFORD LOWERCASE LETTER DATASET
THIRD_DIR = "/home/wshen2011/.cache/kagglehub/datasets/lowercase/"
train_data_3 = pd.read_csv(THIRD_DIR+'letter.data', delimiter='\t', header=None)

test_set = train_data_3.shape[0]//5
for i in range(train_data_3.shape[0]):
    X = np.array(train_data_3.iloc[i,6:-1]).astype(float).reshape(16,8) * 255
    X = np.pad(X, pad_width=((6, 6), (10, 10)), mode='constant', constant_values=0).flatten()
    # y_train = np.array(train_data.iloc[:,0].values)
    Y = np.array([convert_lowercase_classification(train_data_3.iloc[i,1])])

    if i < test_set:
        with open("letter_test.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.concatenate([Y,X]))
    else:
        with open("letter_train.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.concatenate([Y,X]))

subprocess.run(['shuf', "letter_train.csv", '-o', "letter_train_shuffle.csv"])

result = subprocess.run(['wc', '-l', "letter_train_shuffle.csv"], capture_output=True, text=True)
print(result.stdout)
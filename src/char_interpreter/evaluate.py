import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = "/home/wshen2011/.cache/kagglehub/datasets/crawford/emnist/versions/3/"
test_data = pd.read_csv(BASE_DIR+'emnist-bymerge-test.csv')
label_map = pd.read_csv(BASE_DIR+'emnist-bymerge-mapping.txt', sep='\s+', header=None)
# create a dictionary that maps each index to its corresponding ASCII character. 
mapping_dict = {row[0]: chr(row[1]) for _, row in label_map.iterrows()}

x_test = np.array(test_data.iloc[:,1:].values)
y_test = np.array(test_data.iloc[:,0].values)
y_test = tf.keras.utils.to_categorical(y_test -1, len(mapping_dict))

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

x_test = np.apply_along_axis(rotate, 1, x_test)

# fig, axes = plt.subplots(3,5,figsize=(10,8))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(x_test[i].reshape([28,28]))
# plt.show()

x_test = x_test.reshape(-1, 28, 28, 1)

# Load the model architecture and weights together
model = tf.keras.models.load_model('models/emnist_bymerge_2024-11-10 02:43:30.609496.keras')
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

test_predictions = model.predict(x_test)
y_pred_classes = np.argmax(test_predictions, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


fig, axes = plt.subplots(3,5,figsize=(10,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape([28,28]))
    ax.set_title(f'Prediction {mapping_dict[y_pred_classes[i]]},\nTrue {mapping_dict[y_true_classes[i]]}')
plt.show()

cm = confusion_matrix(y_true_classes, y_pred_classes)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in mapping_dict.values()])
disp.plot(cmap=plt.cm.Blues)
plt.rcParams.update({'font.size': 10})  # Increase the font size for better readability


plt.title('Confusion Matrix for EMNIST')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
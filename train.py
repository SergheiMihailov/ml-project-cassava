import pandas as pd
import json, os, datetime
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def train_model(selected_architecture, architecture_name, train_set, val_set):
    model = tf.keras.Model(inputs, selected_architecture(inputs))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    
    logdir = os.path.join("logs", architecture_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
            train_set,
            steps_per_epoch=train_set.n // 32,
            epochs=30,
            validation_data=val_set,
            validation_steps=val_set.n // 32,
            callbacks=[tensorboard_callback])

    model.save(os.path.join("models", architecture_name))

    test_pred_raw = model.predict(val_set)
    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = sklearn.metrics.confusion_matrix(val_set.classes, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=[0,1,2,3,4])
    figure.savefig(f'/logs/images/{architecture_name}.png')


data = pd.read_csv('train.csv')
f = open('label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}
data['class_name'] = data.label.map(real_labels)


train_path = 'train_images/'
train,val = train_test_split(data, test_size = 0.10, random_state = 42, stratify = data['class_name'])

IMG_SIZE = 512
SIZE = (IMG_SIZE,IMG_SIZE)
N_CLASSES = 5
BATCH_SIZE = 16
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
dataget = ImageDataGenerator()

train_set = dataget.flow_from_dataframe(train,
                                directory = train_path,
                                x_col = 'image_id',
                                y_col = 'class_name',
                                target_size = SIZE,
                                color_mode="rgb",
                                class_mode = 'categorical',
                                batch_size = BATCH_SIZE)

val_set = dataget.flow_from_dataframe(val,
                                directory = train_path,
                                x_col = 'image_id',
                                y_col = 'class_name',
                                target_size = SIZE,
                                color_mode="rgb",
                                class_mode = 'categorical',
                                batch_size = BATCH_SIZE)


inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

efficientNetB0 = tf.keras.applications.EfficientNetB0(
    include_top=True, weights=None, input_tensor=None,
    input_shape=INPUT_SHAPE, pooling=None, classes=N_CLASSES,
    classifier_activation='softmax', drop_connect_rate=0.4
)

mobileNetV3Small = tf.keras.applications.MobileNetV3Small(
    input_shape=INPUT_SHAPE, alpha=0.7, minimalistic=True, include_top=True,
    weights=None, input_tensor=None, classes=N_CLASSES, pooling=None,
    dropout_rate=0.2, classifier_activation='softmax'
)

resNet50V2 = tf.keras.applications.ResNet50V2(
    include_top=True, weights=None, input_tensor=None,
    input_shape=INPUT_SHAPE, pooling=None, classes=N_CLASSES,
    classifier_activation='softmax'
)

train_model(efficientNetB0, "efficientNetB0", train_set, val_set)
train_model(mobileNetV3Small, "mobileNetV3Small", train_set, val_set)
train_model(resNet50V2, "resNet50V2", train_set, val_set)

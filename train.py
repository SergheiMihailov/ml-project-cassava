import pandas as pd
import json
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import callbacks.TensorBoard

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



selected_architecture = mobileNetV3Small

model = tf.keras.Model(inputs, selected_architecture(inputs))
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

epochs = 10 

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(
        train_set,
        steps_per_epoch=train_set.n // 32,
        epochs=30,
        validation_data=val_set,
        validation_steps=val_set.n // 32)
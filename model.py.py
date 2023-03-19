# -*- coding: utf-8 -*-
"""Cats-Vs-Dogs model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IxUfkAE-0ZeWB2t95k_kpsIrEW1wfHHZ
"""

# Dataset - https://www.kaggle.com/datasets/salader/dogs-vs-cats

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import zipfile
zip_ref = zipfile.ZipFile('/content/drive/MyDrive/Datasets/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# extracting train file
zip_ref = zipfile.ZipFile('/content/train.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# extracting test file
zip_ref = zipfile.ZipFile('/content/test1.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

"""Defining the image properties"""

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

"""Importing the dataset"""

filenames = os.listdir('/content/train')

categories=[]
for f_name in filenames:
    category=f_name.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filename':filenames,
                   'category':categories})

df.head()

"""Managing data"""

df["category"] = df["category"].replace({0:'cat',1:'dog'})
train_df,validate_df = train_test_split(df,test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=32

train_df.head()

validate_df.head()

"""Test data preparation"""

test_filenames = os.listdir("/content/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

nb_samples

"""Training and validation data generator"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 30

def generators(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True, 
        validation_split = 0.1,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_dataframe(
        train_df,
        "/content/train",
        target_size = (height, width), 
        #classes = (0,1),
        x_col='filename',y_col='category',
        class_mode='categorical',
        batch_size = batch_size,
        #subset = 'training', 
    )

    val_dataset = imgdatagen.flow_from_dataframe(
        validate_df,
        "/content/train",
        target_size = (height, width), 
        #classes = (0,1),
        x_col='filename',y_col='category',
        class_mode='categorical',
        batch_size = batch_size,
        #subset = 'validation'
     )
    
    test_dataset = imgdatagen.flow_from_dataframe(test_df,
                                                 "/content/test1",x_col='filename',y_col=None,
                                                 target_size=(height, width),
                                                 class_mode=None,
                                                 batch_size=batch_size)
    return train_dataset, val_dataset, test_dataset

!pip install keras-resnet

from keras_resnet.models import ResNet50

from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

from tensorflow.keras.utils import load_img, img_to_array

resnet50 = keras.applications.resnet50
train_dataset, val_dataset, test_dataset = generators((224,224), preprocessing=resnet50.preprocess_input)

pre_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in pre_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(pre_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
full_model = keras.models.Model(inputs=pre_model.input, outputs=predictions)

full_model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  metrics=['acc'])

history11 = full_model.fit_generator(
    train_dataset, 
    workers=10,
    epochs=5,
    validation_data = val_dataset,
)

"""saving model"""

full_model.save('resnet50.h5')

"""visualizing"""

def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    
    plt.show()

plot_history(history11, yrange=(0.9,1))

"""predicting test file"""

predict = full_model.predict_generator(test_dataset, steps=np.ceil(nb_samples/batch_size))

"""Convert labels to categories"""

#test_df['category'] = np.argmax(predict, axis=-1)

#label_map = dict((v,k) for k,v in train_generator.class_indices.items())
#test_df['category'] = test_df['category'].replace(label_map)

#test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

"""prediction on custom data"""

img_path = '/content/drive/MyDrive/Lucy /IMG_3674.JPG'
img = load_img(img_path, target_size=(224,224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = full_model.predict(x)[0]
th = np.argmax(preds)
print(preds)

if preds[0] > 0.5:
  print("Prediction: cat")
elif preds[1] > 0.5:
  print("Prediction: dog")
else:
  print("Prediction: Other")
  
plt.imshow(img)

"""for render"""

!pip freeze > requirements.txt

!pip install -r requirements.txt

import sys
print(sys.version)
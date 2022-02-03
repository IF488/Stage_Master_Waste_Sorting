# -*- coding: utf-8 -*-
"""trashnet_mobileNetipynb

Automatically generated by Colaboratory.

Author: Ishan FOOLELL
Description: Algorithm to train MobileNetv2 model with trashnet dataset
Course: Master Artificial Intelligence and Robotics
University: Universite Des Mascareignes
"""

import numpy as np
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D
from tensorflow.keras.models  import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt

train_aug = ImageDataGenerator(shear_range=0.1,
                               vertical_flip=True,
                               horizontal_flip=True,
                               validation_split=0.1,
                               zoom_range=0.3,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1)
                               

train_set=train_aug.flow_from_directory('/content/drive/MyDrive/dataset-resized',
                                    target_size=(312,312),
                                    batch_size=32,
                                    class_mode='categorical',
                                    subset='training')

labels = (train_set.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)

test=ImageDataGenerator(validation_split=0.2)                      

test_set=test.flow_from_directory('/content/drive/MyDrive/dataset-resized',
                                        target_size=(312,312),
                                        batch_size=32,
                                        class_mode='categorical',
                                        subset='validation')

for image_batch, label_batch in train_set:
  break
image_batch.shape, label_batch.shape

"""### Writing the labels file"""

print (train_set.class_indices)

Labels = '\n'.join(sorted(train_set.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(Labels)

"""## Building CNN"""

import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(input_shape=(312,312,3),
                                               include_top=False,
                                               weights='imagenet')

base_model

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(6,activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

preprocess_input

inputs = tf.keras.Input(shape=(312, 312, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

loss0, accuracy0 = model.evaluate(test_set)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

EPOCHS = 10
history = model.fit(train_set,
                    epochs=EPOCHS,
                    validation_data=test_set)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS =  EPOCHS + FINE_TUNE_EPOCHS

history_fine = model.fit(train_set,
                         epochs=TOTAL_EPOCHS,
                         initial_epoch=history.epoch[-1],
                         validation_data=test_set)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
#loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(test_set)

print('Test loss :', loss)
print('Test accuracy :', accuracy)

from tensorflow.keras.preprocessing import image_dataset_from_directory

validation_dataset = image_dataset_from_directory('/content/drive/MyDrive/data/valid',
                                                  shuffle=True,
                                                  batch_size=32,
                                                  image_size=(312,312))

from tensorflow.keras.preprocessing import image

img_path = '/content/drive/MyDrive/data/test/metal32.jpg'

img_V = image.load_img(img_path, target_size=(312,312))
#img = image.img_to_array(img, dtype=np.uint8)
#img = np.array(img)/255.0

img = tf.image.decode_jpeg(tf.io.read_file(img_path),channels=3)

img = tf.image.resize(img,[312, 312])

plt.title("Loaded Image")
plt.axis('off')
plt.imshow(img_V)

p = model.predict(img[np.newaxis, ...])
#p = model.predict(img)


print(img.shape)

classes = np.argmax(p, axis = 1)
print(classes)

#print("Predicted shape",p.shape)
print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:",predicted_class)

model.save('/content/drive/MyDrive/StaZ/mobilenet_model_v2.h5')
print('Model Saved!')

file = "Mob_Garbage_v2.h5"
tf.keras.models.save_model(model,file)
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(file)
tflite_model=converter.convert()
open("Mob_garbage_v2.tflite",'wb').write(tflite_model)

classes=[]
prob=[]
print("\n-------------------Individual Probability--------------------------------\n")

for i,j in enumerate (p[0],0):
    print(labels[i].upper(),':',round(j*100,2),'%')
    classes.append(labels[i])
    prob.append(round(j*100,2))
    
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(classes))
    plt.bar(index, prob)
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(index, classes, fontsize=12, rotation=20)
    plt.title('Probability for loaded image')
    plt.show()
plot_bar_x()

"""## Accuracy Graph"""

!tensorboard dev upload \
  --logdir logs \
  --name "Sample op-level graph" \
  --one_shot

for image_test, label_test in test_set:
  break
#label_test
y_label = np.argmax(label_test, axis = -1)
y_label

y_pred = model.predict(image_test)
classes = np.argmax(y_pred, axis = -1)
classes

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_label, classes)
print('Confusion Matrix\n')
print(confusion)

import seaborn as sns
sns.heatmap(confusion, annot=True, cmap='Blues')
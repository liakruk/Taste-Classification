import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import os
from tensorflow.keras.optimizers import RMSprop
from  keras.layers import Input
import pandas as pd
import io
from math import log
import numpy as np
from keras.regularizers import L1, L2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(tf.__version__)

train_dataset = pd.read_csv('train.csv')
x_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
print(y_train)

validation_dataset = pd.read_csv('validation.csv')
x_val = validation_dataset.iloc[:, :-1].values
y_val = validation_dataset.iloc[:, -1].values
print(y_val)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((x_train.shape[1],)),
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.15),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(32, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(16, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.05),
	tf.keras.layers.Dense(2, activation='softmax')
	])

class_weights = compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
for i in range(len(class_weights)):
	class_weights[i] = min(class_weights[i],4)
# class_weights[0] = 1.2
# class_weights[1] = 0.8
print(class_weights)
class_weight_dict = dict(enumerate(class_weights))

num_epochs = 2000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
				loss='sparse_categorical_crossentropy',
				metrics = ['accuracy'])

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.1,
        patience = 30,
        verbose  = 1,
        mode     = 'min',
        min_delta  = 0.001,
        cooldown = 0,
        restore_best_weights=True,
        min_lr   = 0
    )

history = model.fit(x_train,y_train,
					validation_data=(x_val,y_val),
					# batch_size=512,
					# steps_per_epoch=512,
					epochs = num_epochs,
					# validation_steps=128,
					verbose = 1,
					class_weight=class_weight_dict,
					callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor='loss',patience=80,min_delta=0.001,restore_best_weights=False),
					reduce_on_plateau
				])
model.save('dnn_model.h5')

model.evaluate(x_val,y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


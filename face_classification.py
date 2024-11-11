"""
Apply pretrained model to generate the classification model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from keras.applications import resnet

# Data Path
base_dir = 'train_dataset'

# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # validation_size = 0.2
)

# Split train and test data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # Change input size
    batch_size=32,
    class_mode='categorical',
    subset='training'  # train data
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # validation data
)


# Use MobileNet / ResNet101 ...
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Activate the latest 7 layers
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-7:]:
    layer.trainable = True

# Add top layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
# Add dropout layers
x = layers.Dropout(0.5)(x)

x = layers.Dense(3, activation='softmax')(x)

# Model
model = models.Model(inputs=base_model.input, outputs=x)

# Model compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30  # Epoch
)

# Save model
model.save(r'E:\pythonproject\pythonProject\pythonproject\Face\model\classifier_elon_zurk.h5')


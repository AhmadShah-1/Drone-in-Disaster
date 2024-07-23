import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import os
import datetime

# Ensure you are using TensorFlow's Keras
print("Tensorflow version:", tf.__version__)

# Load the dataset
train_dataset_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Datasets/Images/Emotion_Classification/train'
test_dataset_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Datasets/Images/Emotion_Classification/test'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',  # For multiple classes
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',  # For multiple classes
    subset='validation'
)

# Build the model
model = Sequential([
    Rescaling(1./255, input_shape=(48, 48, 3)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(7, activation='softmax')  # For 7 emotion classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()

# Set the log directory for TensorBoard
log_dir = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Training/Emotion_Classification/Version1/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Callbacks for early stopping and model checkpoint
best_model_filepath = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Training/Emotion_Classification/Version1/Models/best_model.keras'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(best_model_filepath, monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[model_checkpoint, early_stopping, tensorboard_callback]
)

# Plot training & validation accuracy and loss values
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

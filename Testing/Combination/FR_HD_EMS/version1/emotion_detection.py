import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Define the class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



# Function to preprocess the grayscale images
def preprocess_image(image):
    # Resize the image to 48x48
    image = cv2.resize(image, (48, 48))
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the emotion
def predict_emotion(model, image):
    # Preprocess the image
    image = preprocess_image(image)
    # Make the prediction
    predictions = model.predict(image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name


def emotion_detection(queue, emotion_results):
    model = load_model('best_model.keras')

    while True:
        if not queue.empty():
            item = queue.get()
            print("Queue not empty")

            if item[0] == 'process_emotion':
                print("Emotion Processing")
                gray_face, tracker_id = item[1:]

                # Predict emotion
                predicted_emotion = predict_emotion(model, gray_face)
                emotion_results[tracker_id] = predicted_emotion

            elif item[0] == 'update_tracker':
                tracker_id, emotion = item[1:]
                emotion_results[tracker_id] = emotion

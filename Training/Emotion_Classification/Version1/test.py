import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the saved model
model = load_model('best_model.keras')

# Define the class names (adjust according to your dataset)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess the images
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
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
def predict_emotion(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    # Make the prediction
    predictions = model.predict(image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Test the model on custom images
image_paths = [
    'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Datasets/Output_Images/Emotion_Classification/test/angry/PrivateTest_1054527.jpg',
    # Add more image paths as needed
]

for image_path in image_paths:
    predicted_emotion = predict_emotion(image_path)
    print(f"Image: {image_path}, Predicted Emotion: {predicted_emotion}")
    # Display the image with the predicted emotion
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Predicted Emotion: {predicted_emotion}")
    plt.axis('off')
    plt.show()

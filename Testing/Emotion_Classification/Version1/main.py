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

# Function to process a directory of images
def process_directory(model, directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            predicted_emotion = predict_emotion(model, image)
            save_image_with_emotion(image, predicted_emotion, output_directory, filename)

# Function to save the image with the predicted emotion
def save_image_with_emotion(image, emotion, output_directory, filename):
    # Create the output file path
    output_path = os.path.join(output_directory, filename)

    # Add the predicted emotion as a label on the image
    labeled_image = image.copy()
    cv2.putText(labeled_image, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the labeled image
    cv2.imwrite(output_path, labeled_image)

    # Display the image with the predicted emotion
    plt.figure()
    plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
    plt.title(emotion)
    plt.axis('off')
    plt.show()

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

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model('C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Combination/FR_HD_EMS/version1/best_model.keras')

    # Path to the directory containing images
    directory_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Emotion_Classification/Version1/Images'
    output_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Emotion_Classification/Version1/Output_Images'

    # Process the directory of images
    process_directory(model, directory_path, output_directory)


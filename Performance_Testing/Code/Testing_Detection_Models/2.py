import os
import cv2
import csv
import time
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the input and output directories
input_dir = "H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/DroneView/Combination"  # Path to the directory containing images
output_dir = "H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/Detections/Combination"  # Path to save the images with detections

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Specify the YOLOv8 models to test (e.g., YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
yolo_models = [
    'yolov8n',  # Nano model (smallest and fastest)
    'yolov8s',  # Small model
    'yolov8m',  # Medium model
    'yolov8l',  # Large model
    'yolov8x',  # Extra-large model (most accurate but slowest)
]

# Load the YOLOv8 models
yolo_models = [YOLO(f'{model}.pt') for model in yolo_models]  # Replace with your specific model paths if necessary

# CSV file path to save detection results including inference time
csv_file_path = os.path.join(output_dir, 'yolo_detections.csv')

# Create the CSV file and write the header if it doesn't exist
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(
        ['Model', 'Image', 'Number of Person Detections', 'Average Confidence Score', 'Inference Time (ms)'])


def detect_and_save_images(model, model_name, input_directory, output_directory, csv_writer):
    """
    Detects objects in images using the specified YOLOv8 model and saves the resulting images.

    :param model: The YOLO model to use for detection.
    :param model_name: The name of the YOLO model.
    :param input_directory: Directory containing input images.
    :param output_directory: Directory to save detected images.
    :param csv_writer: CSV writer object to write detection results.
    """
    # Create a subdirectory for this model
    model_output_dir = os.path.join(output_directory, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Iterate through each image in the input directory
    for image_file in os.listdir(input_directory):
        # Full path to the image
        image_path = os.path.join(input_directory, image_file)

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Ensure the image was read correctly
        if img is None:
            print(f"Skipping file {image_file}: not a valid image.")
            continue

        # Measure inference time
        start_time = time.time()
        results = model(img)  # Perform object detection with YOLOv8 model
        end_time = time.time()

        # Calculate inference time in milliseconds
        inference_time = (end_time - start_time) * 1000  # Convert seconds to milliseconds

        # Filter detections to only include "person" class (Assuming class index 0 is "person")
        person_detections = [det for det in results[0].boxes if det.cls == 0]  # Adjust the class index if necessary

        # Calculate the number of person detections
        num_person_detections = len(person_detections)

        # Calculate the average confidence score for person detections
        if num_person_detections > 0:
            avg_confidence_score = sum(det.conf.item() for det in person_detections) / num_person_detections
        else:
            avg_confidence_score = 0.0

        # Render the results on the image (results are stored in results[0].plot())
        detected_image = results[0].plot()

        # Save the image with detections to the model's output directory
        output_path = os.path.join(model_output_dir, image_file)
        cv2.imwrite(output_path, detected_image)

        print(
            f"Processed {image_file} with model {model_name}. Inference time: {inference_time:.2f} ms. Saved to {output_path}.")

        # Write the results to the CSV file
        csv_writer.writerow(
            [model_name, image_file, num_person_detections, round(avg_confidence_score, 2), round(inference_time, 2)])


# Open the CSV file for appending detection results
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each YOLOv8 model and perform detection
    for yolo_model, model_name in zip(yolo_models, ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']):
        print(f"Using model: {model_name}")
        detect_and_save_images(yolo_model, model_name, input_dir, output_dir, csv_writer)

# Plotting Inference Time Results
# Load the updated CSV with inference times
data = pd.read_csv(csv_file_path)

# Set up the plotting style
sns.set(style="whitegrid")

# Create a bar chart for the average inference time per model
plt.figure(figsize=(10, 6))
avg_inference_time = data.groupby('Model')['Inference Time (ms)'].mean().reset_index()
sns.barplot(x='Model', y='Inference Time (ms)', data=avg_inference_time, palette="coolwarm")
plt.title('Average Inference Time per Model (ms)')
plt.ylabel('Inference Time (ms)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "avg_inference_time_per_model.png"))
plt.show()

import os
import cv2
import csv
from ultralytics import YOLO
import shutil

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

# CSV file path to save detection results
csv_file_path = os.path.join(output_dir, 'yolo_detections.csv')

# Create the CSV file and write the header
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(['Model', 'Image', 'Number of Person Detections', 'Average Confidence Score'])


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

        # Perform object detection with YOLOv8 model
        results = model(img)

        # Filter detections to only include "person" class
        person_detections = [det for det in results[0].boxes if
                             det.cls == 0]  # Assuming class 0 is 'person' in the model

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

        print(f"Processed {image_file} with model {model_name}. Saved to {output_path}.")

        # Write the results to the CSV file
        csv_writer.writerow([model_name, image_file, num_person_detections, round(avg_confidence_score, 2)])


# Open the CSV file for appending detection results
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each YOLOv8 model and perform detection
    for yolo_model, model_name in zip(yolo_models, ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']):
        print(f"Using model: {model_name}")
        detect_and_save_images(yolo_model, model_name, input_dir, output_dir, csv_writer)
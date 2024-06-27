# This version processes a frame, finds bounding boxes of individuals,
# Finds faces in bounding boxes, then assign ID if face does not exist in directory, otherwise reassign ID


import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import face_recognition
import os
from PIL import Image

# Directory to save unique faces
faces_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Combination/FR_and_HD/Detected_Faces'
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLO model and move to GPU
model = YOLO('C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/weights/yolov8s.pt').to('cuda')

# Initialize ByteTrack and Annotators from supervision
byte_tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
VIDEO_PATH = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Input_Video/nyc_people_walking1.mp4"

def is_face_already_detected(face_image, directory):
    face_encoding = face_recognition.face_encodings(face_image)
    if not face_encoding:
        return False, None
    face_encoding = face_encoding[0]

    for file_name in os.listdir(directory):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            known_image = face_recognition.load_image_file(os.path.join(directory, file_name))
            known_encoding = face_recognition.face_encodings(known_image)
            if known_encoding:
                known_encoding = known_encoding[0]
                results = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                if results[0]:
                    return True, file_name
    return False, None

# Define the callback function
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Run YOLO model on frame
    results = model(frame)[0]
    # Transform the detection results ([0]) to a form that sv can use
    detections = sv.Detections.from_ultralytics(results)

    # Initialize an empty list to store indices of human detections
    human_indices = []

    # Loop through each detection and its corresponding class_id
    for i, class_id in enumerate(detections.class_id):
        # Check if the class_id corresponds to a human (assuming 0 is the class_id for humans)
        if class_id == 0:
            # If it is a human, add the index to the human_indices list
            human_indices.append(i)

    # Create a new Detections object containing only human detections
    human_detections = sv.Detections(
        xyxy=detections.xyxy[human_indices],
        confidence=detections.confidence[human_indices],
        class_id=detections.class_id[human_indices],
        tracker_id=detections.tracker_id[human_indices] if detections.tracker_id is not None else None,
        mask=detections.mask[human_indices] if detections.mask is not None else None,
        data={k: v[human_indices] for k, v in detections.data.items()} if detections.data else None
    )

    # updates the tracker with new detections
    tracked_detections = byte_tracker.update_with_detections(human_detections)

    # Debugging: Print the structure of tracked_detections
    print(f"Tracked Detections: {tracked_detections}")

    # Track the tracker ID, class name, and confidence score
    # Initialize an empty list to store the labels
    labels = []
    iterate = 0

    # Iterate over the tracked detections' properties using the zip function
    for xyxy, confidence, class_id, tracker_id in zip(
            tracked_detections.xyxy,
            tracked_detections.confidence,
            tracked_detections.class_id,
            tracked_detections.tracker_id
    ):
        # Format the label using an f-string
        label = f"# {tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        # Add the formatted label to the labels list
        labels.append(label)


        #FROMHERE
        # Extract the part of the image within the bounding box
        x1, y1, x2, y2 = map(int, xyxy)
        person_image = frame[y1:y2, x1:x2]
        gray_person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_person_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # Iterate over the detected faces and save unique faces
        for j, (x, y, w, h) in enumerate(faces):
            print("Iterate", iterate)
            iterate += 1
            # Extract the face from the image
            face = person_image[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Check if the face is already detected
            is_detected, file_name = is_face_already_detected(face_rgb, faces_directory)
            if not is_detected:
                # Convert to PIL Image format
                face_image = Image.fromarray(face_rgb)

                # Save the unique face image
                face_image_path = os.path.join(faces_directory, f'{tracker_id}.jpg')
                face_image.save(face_image_path)
                print(f'Saved unique face frame_{index}_face_{tracker_id}_{j + 1} to {face_image_path}')
            else:
                print(f'Face frame_{index}_face_{tracker_id}_{j + 1} is already detected as {file_name}.')
        #TOHERE

    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

    # Display the annotated frame
    cv2.imshow('Processed Video', annotated_frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None  # Return None to stop the video processing

    return annotated_frame

# Process the video
sv.process_video(source_path=VIDEO_PATH, target_path="../../Output_Video/Testing2.mp4", callback=callback)

# Release the video window
cv2.destroyAllWindows()

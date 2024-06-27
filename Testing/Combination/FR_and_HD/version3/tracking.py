import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import os
from multiprocessing import Value, Queue, Manager

# Directory to save unique faces
faces_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Combination/FR_and_HD/Detected_Faces'
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)

# Load YOLO model and move to GPU
model = YOLO('C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/weights/yolov8s.pt').to('cuda')

# Initialize ByteTrack and Annotators from supervision
byte_tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def tracking(VIDEO_PATH, queue, temp_id_counter, permanent_id_counter, temporary_ids):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # Update the tracker with new detections
        tracked_detections = byte_tracker.update_with_detections(human_detections)

        # Debugging: Print the structure of tracked_detections
        print(f"Tracked Detections: {tracked_detections}")

        # Initialize an empty list to store the labels
        labels = []

        # Iterate over the tracked detections' properties using the zip function
        for xyxy, confidence, class_id, tracker_id in zip(
                tracked_detections.xyxy,
                tracked_detections.confidence,
                tracked_detections.class_id,
                tracked_detections.tracker_id
        ):
            # If the tracker ID already has a permanent ID, skip processing
            if tracker_id in temporary_ids and not isinstance(temporary_ids[tracker_id], str):
                label = f"# {temporary_ids[tracker_id]} {model.model.names[class_id]} {confidence:0.2f}"
                labels.append(label)
                continue

            # Assign a temporary ID if the tracker ID is not recognized
            if tracker_id not in temporary_ids:
                with temp_id_counter.get_lock():
                    temporary_ids[tracker_id] = f'Temp_{temp_id_counter.value}'
                    temp_id_counter.value += 1

            # Check if the tracker ID is labeled with a temporary ID
            if isinstance(temporary_ids[tracker_id], str) and 'Temp' in temporary_ids[tracker_id]:
                # Extract the part of the image within the bounding box
                x1, y1, x2, y2 = map(int, xyxy)
                person_image = frame[y1:y2, x1:x2]
                gray_person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray_person_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

                # Iterate over the detected faces and send them to the processing queue
                for j, (x, y, w, h) in enumerate(faces):
                    # Extract the face from the image
                    face = person_image[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    # Send face to the queue for processing
                    queue.put(('process_face', face_rgb, tracker_id, index, j))

            # Format the label using an f-string
            label = f"# {temporary_ids[tracker_id]} {model.model.names[class_id]} {confidence:0.2f}"
            # Add the formatted label to the labels list
            labels.append(label)

        # Ensure that every detection has a label
        if len(labels) != len(tracked_detections.xyxy):
            raise ValueError("The number of labels provided does not match the number of detections. Each detection should have a corresponding label.")

        annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        # Display the annotated frame
        cv2.imshow('Processed Video', annotated_frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None  # Return None to stop the video processing

        return annotated_frame

    # Process the video
    sv.process_video(source_path=VIDEO_PATH, target_path="../../Output_Video/Tracking.mp4", callback=callback)

    # Release the video window
    cv2.destroyAllWindows()

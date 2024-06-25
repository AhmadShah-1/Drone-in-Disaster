import cv2
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
from loguru import logger  # Ensure loguru is imported

# Simple class to hold configuration arguments
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False

# Load YOLOv8 model
model = YOLO('weights/yolov8s.pt').to('cuda')  # Move model to GPU

# Initialize BYTETracker with configuration arguments
tracker_args = TrackerArgs()
tracker = BYTETracker(tracker_args)

# Initialize video capture
cap = cv2.VideoCapture('C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Input_Video/output_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Extract bounding boxes and class ids
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # Filter for person class (usually class id 0, but verify for your model)
    person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]

    # Convert boxes to xywh format required by BYTETracker
    person_boxes_xywh = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for x1, y1, x2, y2 in person_boxes])

    # Concatenate boxes and scores for tracker.update
    output_results = np.hstack((person_boxes_xywh, person_scores[:, np.newaxis]))

    # Prepare image info and size for tracker.update method
    img_info = frame.shape[:2]  # height, width
    img_size = frame.shape[:2]  # height, width

    # Ensure output_results is a NumPy array on the CPU
    output_results = torch.tensor(output_results).cpu().numpy()

    # Update tracker with detections
    outputs = tracker.update(output_results, img_info, img_size)

    print(outputs)
    input1 = input("proceed?")

    # Draw bounding boxes and tracker IDs
    for track in outputs:
        print("processed")
        tlwh = track.tlwh
        track_id = track.track_id
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected and tracked bounding boxes
    cv2.imshow('YOLOv8 Detection and ByteTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

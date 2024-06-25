import cv2
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
from loguru import logger  # Ensure loguru is imported

# Verify CUDA
assert torch.cuda.is_available(), "CUDA is not available. Ensure you have installed the CUDA-enabled PyTorch."

# Load YOLOv8 model
model = YOLO('yolov8s.pt').to('cuda')  # Move model to GPU

# Initialize BYTETracker
tracker = BYTETracker()

# Initialize video capture
cap = cv2.VideoCapture('path_to_your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the frame
    results = model(frame, device='cuda')

    # Extract bounding boxes and class ids
    boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
    scores = results.conf[0].cpu().numpy()  # Confidence scores
    class_ids = results.cls[0].cpu().numpy()  # Class IDs

    # Filter for person class (usually class id 0, but verify for your model)
    person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]

    # Convert boxes to xywh format required by BYTETracker
    person_boxes_xywh = np.array([[(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1] for x1, y1, x2, y2 in person_boxes])

    # Update tracker with detections
    outputs = tracker.update(person_boxes_xywh, person_scores, frame)

    # Draw bounding boxes and tracker IDs
    for output in outputs:
        x1, y1, x2, y2, track_id = map(int, output[:4]) + [int(output[4])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected and tracked bounding boxes
    cv2.imshow('YOLOv8 Detection and ByteTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

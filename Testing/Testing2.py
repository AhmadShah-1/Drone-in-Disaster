import supervision as sv
from ultralytics import YOLO
import numpy as np


# Load YOLO model and move to GPU
model = YOLO('weights/yolov8s.pt').to('cuda')

# Initialize ByteTrack and BoxAnnotator from supervision
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()
VIDEO_PATH = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Input_Video/nyc_people_walking1.mp4"


# Define the callback function
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Run YOLO model on frame
    results = model(frame)[0]
    # Transform the detection results ([0]) to a form that sv can use
    detections = sv.Detections.from_ultralytics(results)
    # updates the tracker with new detections
    tracked_detections = byte_tracker.update_with_detections(detections)

    # Debugging: Print the structure of tracked_detections
    print(f"Tracked Detections: {tracked_detections}")

    # Track the tracker ID, class name, and confidence score
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for xyxy, confidence, class_id, tracker_id in zip(
            tracked_detections.xyxy,
            tracked_detections.confidence,
            tracked_detections.class_id,
            tracked_detections.tracker_id
        )
    ]


    # Annotate the information of the bounding boxes and labels
    return annotator.annotate(scene=frame.copy(), detections=tracked_detections, labels=labels)


# Process the video
sv.process_video(source_path=VIDEO_PATH, target_path="../Output_Video/Testing2.mp4", callback=callback)

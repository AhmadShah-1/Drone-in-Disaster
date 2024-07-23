import cv2
import os
from PIL import Image
import face_recognition
import numpy as np

# Load the image
image_path = '/Testing/Face_Recognition/Output_Images/People/image2.jpg'
image = cv2.imread(image_path)

# Load the deep learning model for face detection
prototxt_path = '/Testing/Face_Recognition/Output_Images/weights/deploy.prototxt'  # Path to the deploy.prototxt.txt
model_path = '/Testing/Face_Recognition/Output_Images/weights/res10_300x300_ssd_iter_140000.caffemodel'  # Path to the .caffemodel
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Prepare the image for the model
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Detect faces
net.setInput(blob)
detections = net.forward()

# Directory to save unique faces
faces_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Face_Recognition/Output_Images/Detected_Faces/'
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)


def is_face_already_detected(face_image, directory):
    face_encoding = face_recognition.face_encodings(face_image)
    if not face_encoding:
        return False
    face_encoding = face_encoding[0]

    for file_name in os.listdir(directory):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            known_image = face_recognition.load_image_file(os.path.join(directory, file_name))
            known_encoding = face_recognition.face_encodings(known_image)
            if known_encoding:
                known_encoding = known_encoding[0]
                results = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                if results[0]:
                    return True
    return False


# Iterate over the detected faces and save unique faces
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Confidence threshold
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        # Extract the face from the image
        face = image[y:y1, x:x1]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Check if the face is already detected
        if not is_face_already_detected(face_rgb, faces_directory):
            # Convert to PIL Image format
            face_image = Image.fromarray(face_rgb)

            # Save the unique face image
            face_image_path = os.path.join(faces_directory, f'face_{i + 1}.jpg')
            face_image.save(face_image_path)
            print(f'Saved unique face {i + 1} to {face_image_path}')
        else:
            print(f'Face {i + 1} is already detected.')

print(f'Detected and saved unique faces.')

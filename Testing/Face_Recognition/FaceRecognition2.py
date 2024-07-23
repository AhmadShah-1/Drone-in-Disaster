import cv2
import os
from PIL import Image
import face_recognition
import numpy as np
import time

# Load the image
image_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Face_Recognition/Images/People/image2.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the timer
start_time = time.time()
# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

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
for i, (x, y, w, h) in enumerate(faces):
    # Extract the face from the image
    face = image[y:y + h, x:x + w]
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

print(f'Detected and saved {len(faces)} unique faces.')
# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Detected and saved unique faces in {elapsed_time:.2f} seconds.')


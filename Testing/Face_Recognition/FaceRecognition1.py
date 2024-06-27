# This program uses Haarcascades to detect faces and save them

import cv2
from PIL import Image

# Load the image
image_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Second_Iteration/Images/People/image2.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

# Iterate over the detected faces and save each face as a separate image
for i, (x, y, w, h) in enumerate(faces):
    # Extract the face from the image
    face = image[y:y + h, x:x + w]

    # Convert to PIL Image format
    face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    # Save the face image
    face_image_path = f'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Second_Iteration/Images/Detected_Faces/face_{i + 1}.jpg'
    face_image.save(face_image_path)
    print(f'Saved face {i + 1} to {face_image_path}')

print(f'Detected and saved {len(faces)} faces.')

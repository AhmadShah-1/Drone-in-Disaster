import face_recognition
from PIL import Image
import os
from multiprocessing import Queue, Value
import cv2

# Directory to save unique faces
faces_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Combination/FR_HD_EMS/version1/Detected_Faces'

# Record of seen individuals' faces and their tracker IDs
seen_faces = {}

# Tracker IDs that have been processed already
tracker_seen = set()

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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

def face_processing(queue, permanent_id_counter, temporary_ids):
    global seen_faces, tracker_seen

    while True:
        if not queue.empty():
            item = queue.get()

            if item[0] == 'process_person':
                person_image, tracker_id, frame_index = item[1:]

                # Skip if the tracker ID has already been processed
                if tracker_id in tracker_seen:
                    continue

                gray_person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray_person_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

                for j, (x, y, w, h) in enumerate(faces):
                    face = person_image[y:y + h, x:x + w]

                    # Send the grayscale face to the queue for emotion detection
                    queue.put(('process_emotion', face, tracker_id))

                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    print("Processing face")
                    # Check if the face is already detected
                    is_detected, file_name = is_face_already_detected(face_rgb, faces_directory)
                    if not is_detected:
                        print("Face not seen before")
                        # Convert to PIL Image format
                        face_image = Image.fromarray(face_rgb)

                        # Save the unique face image
                        with permanent_id_counter.get_lock():
                            face_id = permanent_id_counter.value
                            face_image_path = os.path.join(faces_directory, f'{face_id}.jpg')
                            face_image.save(face_image_path)
                            print(face_image_path)

                            # Compute the face encoding and store it
                            face_encodings = face_recognition.face_encodings(face_rgb)
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                seen_faces[face_id] = face_encoding  # Store the face encoding for the tracker ID
                                queue.put(('update_tracker', tracker_id, face_id))
                                permanent_id_counter.value += 1
                                print(f'Saved unique face frame_{frame_index}_face_{tracker_id}_{j + 1} to {face_image_path}')
                                tracker_seen.add(tracker_id)
                    else:
                        print(f'Face frame_{frame_index}_face_{tracker_id}_{j + 1} is already detected as {file_name}.')

                        # Update the tracker ID with the corresponding permanent ID
                        detected_id = int(file_name.split('.')[0])
                        queue.put(('update_tracker', tracker_id, detected_id))
                        tracker_seen.add(tracker_id)

            elif item[0] == 'update_tracker':
                tracker_id, permanent_id = item[1:]
                temporary_ids[tracker_id] = permanent_id
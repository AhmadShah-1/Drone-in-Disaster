import cv2
import os

# Path to the directory containing images
image_folder = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Datasets/Images/Aerial_Images/VisDrone2019-MOT-val/sequences/uav0000117_02622_v'
# Output video file path
video_name = 'uav0000117_02622_v.mp4'

# Get a list of all images in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()  # Ensure the images are in the correct order

# Read the first image to get the dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the video writer object
video.release()
print(f"Video saved as {video_name}")

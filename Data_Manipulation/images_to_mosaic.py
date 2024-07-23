from PIL import Image
import os


def create_image_mosaic(directory, output_image, tile_size=(100, 100)):
    # Ensure the output image has a valid extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    if not any(output_image.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Output file must have one of the following extensions: {valid_extensions}")

    # List to hold the images
    images = []

    # Load all images from the directory
    for filename in os.listdir(directory):
        if filename.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = img.resize(tile_size)  # Resize images to tile size
            images.append(img)

    # Calculate the number of rows and columns
    num_images = len(images)
    num_columns = int(num_images ** 0.5)
    num_rows = (num_images + num_columns - 1) // num_columns

    # Create the mosaic canvas
    mosaic_width = num_columns * tile_size[0]
    mosaic_height = num_rows * tile_size[1]
    mosaic_image = Image.new('RGB', (mosaic_width, mosaic_height))

    # Paste the images into the mosaic canvas
    for i, img in enumerate(images):
        row = i // num_columns
        col = i % num_columns
        x = col * tile_size[0]
        y = row * tile_size[1]
        mosaic_image.paste(img, (x, y))

    # Save the output mosaic image
    mosaic_image.save(output_image)



# Usage
directory_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Testing/Combination/FR_HD_EMS/version1/Detected_Faces/'
output_image_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Data_Manipulation/image1.jpg'
create_image_mosaic(directory_path, output_image_path)

from PIL import Image, ImageDraw
import os
import numpy as np
from shapely.geometry import Polygon
import glob
import cv2

def extract_objects_from_images(input_dir, output_dir):
    # Create output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Get path to all image files in the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))

    label_counts = dict()

    for image_path in image_paths:
        # Load image
        im = Image.open(image_path).convert('RGBA')
        im_array = np.asarray(im)

        # Find contours from the mask
        # Create a binary mask by removing white areas
        mask_array = np.any(im_array != [255, 255, 255, 255], axis=2).astype(np.uint8)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all the contours
        for contour in contours:
            # Create a polygon from the contour coordinates
            polygon = Polygon(contour)

            # Get label for the object
            label = "object"

            if label not in label_counts:
                label_counts[label] = 0
                os.makedirs(os.path.join(output_dir, label), exist_ok=True)

            # Apply the mask to the image array
            masked_array = np.where(polygon.contains(Point(x, y)), im_array, 0)

            # Convert the masked array back to Image object
            masked_image = Image.fromarray(masked_array, 'RGBA')

            # Crop the image to the bounding box of the polygon
            x_min, y_min, x_max, y_max = polygon.bounds
            cropped_image = masked_image.crop((x_min, y_min, x_max, y_max))

            # Save the cropped image
            cropped_image.save(os.path.join(output_dir, label, f'{label_counts[label]}.png'))
            label_counts[label] += 1


input_dir =  "/ws/dev/Computer-Vision-Synthetic-Data-Generation/data/input/foregrounds/0"
output_dir = "/ws/dev/Computer-Vision-Synthetic-Data-Generation/data/input/foregrounds/result"
extract_objects_from_images(input_dir, output_dir)
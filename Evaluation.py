import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


# Step 1: Read an image
dataset_folder = "Dataset"
input_images_folder = "ground_truths"
output_folder = "output_images"
folders = ["easy", "medium", "hard"]

# convert all ground truths to jpg format
# Loop through each folder
for folder_name in folders:
    ground_folder_path = os.path.join(dataset_folder, input_images_folder, folder_name)

    # Get all image filenames in the folder
    image_filenames = os.listdir(ground_folder_path)

    # Loop through each groundtruth image
    for image_filename in image_filenames:
        # Open the image file
        img_path = os.path.join(ground_folder_path, image_filename)
        img = Image.open(img_path)

        # Convert the image to RGB mode if it's in palette mode (mode 'P')
        if img.mode == "P":
            img = img.convert("RGB")

        # Convert the image to JPG format if it's not already in JPG format
        if img.format != "JPEG":
            # Change the file extension to .jpg
            new_filename = os.path.splitext(image_filename)[0] + ".jpg"
            # Save the image with the new filename and JPEG format
            img.save(os.path.join(ground_folder_path, new_filename), "JPEG")


# Initialize variables to calculate mIoU
total_iou = 0
num_images = 0

# Loop through each folder
for folder_name in folders:
    ground_folder_path = os.path.join(dataset_folder, input_images_folder, folder_name)
    output_folder_path = os.path.join(dataset_folder, output_folder, folder_name)

    # Get all image filenames in the folder
    image_filenames = os.listdir(output_folder_path)

    # Loop through each output and groundtruth image
    for image_filename in image_filenames:
        # Construct the image path
        output_image_path = os.path.join(output_folder_path, image_filename)
        output_image = cv2.imread(output_image_path)

        groundtruth_path = os.path.join(ground_folder_path, image_filename)
        ground_mask = cv2.imread(groundtruth_path)

        # Convert output_image to grayscale and apply thresholding
        output_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        _, output_binary = cv2.threshold(output_gray, 0, 255, cv2.THRESH_BINARY)

        ground = cv2.cvtColor(ground_mask, cv2.COLOR_BGR2GRAY)
        _, ground_binary = cv2.threshold(ground, 100, 255, cv2.THRESH_BINARY)
        ground_truth_binary = 255 - ground_binary       # invert the image


        # Calculate intersection and union
        intersection = np.logical_and(output_binary, ground_truth_binary)
        union = np.logical_or(output_binary, ground_truth_binary)

        # Calculate IoU
        iou = np.sum(intersection) / np.sum(union)

        # Accumulate IoU
        total_iou = total_iou + iou
        num_images = num_images + 1

        print(f"Iou: {iou:.3f}")
        print(f"Total Iou: {total_iou:.3f}")
        print(f"Image: {num_images:.3f}")
        print(" ")

# Calculate mean IoU (mIoU)
miou = total_iou / num_images if num_images > 0 else 0

print(f"MIOU: {miou:.3f}")










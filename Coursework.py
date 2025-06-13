import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Read an image
dataset_folder = "Dataset"
input_images_folder = "input_images"
image_processing_pipeline_folder = "image_proccessing_pipeline"
output_folder = "output_images"
folders = ["easy", "medium", "hard"]

while True:
    # Loop through each folder
    for folder_name in folders:
        folder_path = os.path.join(dataset_folder, input_images_folder, folder_name)

        # Get all image filenames in the folder
        image_filenames = os.listdir(folder_path)

        # Create an image processing pipeline folder
        ipp_folder_path = os.path.join(dataset_folder, image_processing_pipeline_folder, folder_name)
        if not os.path.exists(ipp_folder_path):
            os.makedirs(ipp_folder_path)

        # Create an output folder if it doesn't exist
        output_folder_path = os.path.join(dataset_folder, output_folder, folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Initialize counter for image numbering
        image_counter = 1

        # Loop through each image
        for image_filename in image_filenames:
            # Construct the image path
            image_path = os.path.join(folder_path, image_filename)
            input_image = cv2.imread(image_path)

            # Step 2: Convert to the desired color space (e.g., RGB to HSV)
            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

            # Extract the value channel
            v_channel = hsv[:, :, 2]

            # Step 3: Preprocess the image (e.g. noise reduction)
            median_filtered = cv2.medianBlur(v_channel, 5)  # Apply median filtering
            blurred_image = cv2.GaussianBlur(median_filtered, (5, 5), 0)

            # Step 4: Thresholding/Segmentation
            _, threshold = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Step 5: Binary image processing (e.g., morphology)
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            cleaned_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            # Additional step: Further closing to fill small holes
            closing_kernel = np.ones((20, 20), np.uint8)  # Adjust the kernel size as needed
            final_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, closing_kernel)

            # Save the processing pipeline images
            ipp_image_folder_path = os.path.join(ipp_folder_path, folder_name + '_' + str(image_counter))
            if not os.path.exists(ipp_image_folder_path):
                os.makedirs(ipp_image_folder_path)

            cv2.imwrite(os.path.join(ipp_image_folder_path, "original_image.jpg"), (input_image))
            cv2.imwrite(os.path.join(ipp_image_folder_path, "hsv_image.jpg"), (hsv))
            cv2.imwrite(os.path.join(ipp_image_folder_path, "v_channel_image.jpg"), v_channel)
            cv2.imwrite(os.path.join(ipp_image_folder_path, "median_filtered_image.jpg"), median_filtered)
            cv2.imwrite(os.path.join(ipp_image_folder_path, "threshold_image.jpg"), threshold)
            cv2.imwrite(os.path.join(ipp_image_folder_path, "cleaned_image.jpg"), cleaned_image)
            cv2.imwrite(os.path.join(ipp_image_folder_path, "output_image.jpg"), final_image)

            # Save the final image in a separate output folder
            output_image_path = os.path.join(output_folder_path, image_filename)
            cv2.imwrite(output_image_path, final_image)

            # Increment image counter
            image_counter += 1

    # Ask the user if they want to see more images
    choice = input("Do you want to see the images? (yes/no): ").lower()
    if choice != 'yes':
        break

    # Ask the user for input to select the image
    selected_image_folder = input("Enter the folder name (easy, medium, hard): ")
    selected_image_index = int(input("Enter the image index (1, 2, 3): "))

    # Display the original and final images
    selected_input_folder_path = os.path.join(dataset_folder, input_images_folder, selected_image_folder)
    selected_output_folder_path = os.path.join(dataset_folder, output_folder, selected_image_folder)
    selected_image_filename = os.listdir(selected_output_folder_path)[selected_image_index - 1]

    original_image_path = os.path.join(selected_input_folder_path, selected_image_filename)
    final_image_path = os.path.join(selected_output_folder_path, selected_image_filename)

    original_image = cv2.imread(original_image_path)

    # Convert the image to RGB format
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    final_image = cv2.imread(final_image_path)

    # Display original and final images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(final_image)
    plt.title("Final Image")
    plt.axis('off')

    plt.show()

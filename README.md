# Image Processing Pipeline 
The code, developed using Python, inputs one of the images from the dataset (easy_1, easy_2, easy_3, medium_1, medium_2, medium_3, hard_1, hard_2, hard_3), and outputs a binary image marking regions corresponding to flower material. 

# Environment Setup 
Before running the code, ensure that you have setup the environment properly. The following statements need to be included at the beginning of the code: <br>

  *import cv2* <br>
  *import numpy as np* <br>
  *import matplotlib.pyplot as plt* <br>
  *import os* <br> 

**NOTE:** In the evaluation file, an additional statement needs to be included along with the above:

  *from PIL import Image* 

# Python Files
## (A) Coursework.py 
The primary functionality of the image processing pipeline is implemented in this code. 
### 1. Folder Creation: 
Upon execution, the pipeline automatically creates two folders within the dataset folder: <br>
* **image_processing_pipeline:** This folder stores the images generated during the processing pipeline. <br>
* **output_images:** This folder stores the final processed images. <br>

### 2. Image Processing: 
The pipeline processes the input images, applying various image processing techniques to enhance their quality or extract specific features. _(More detailed explanation given below in the overview section)_

### 3. User Interaction:
* **Image Viewing:**  The user is prompted to view images after the processing is complete.
* **Selection Options:**: If the user chooses to view images, they select the type of image (easy, medium, or hard) and the index (1, 2, or 3) of the image they want to view.
* **Display Images:** The pipeline displays both the original and final images of the selected type and index.
* **Choice to Continue or Exit:** After viewing the images, the user can choose to view more images or exit the program.
  
## (B) Evaluation.py 
We implemented the evaluation code to assess the performance of our image processing pipeline. We used mIoU (mean Intersection over Union) as the metrics to evaluate our code. 

### 1. Calculation Procedure: <br>
* ****IoU Calculation**:** For each processed image, we calculate the IoU (Intersection over Union) score. IoU measures the overlap between the predicted segmentation mask and the ground truth mask. <br>

* **Total IoU Calculation:** After computing IoU for each image, we sum up the IoU scores to obtain the total IoU for all the processed images.<br>

* **mIoU Calculation:** The mIoU is then calculated by dividing the total IoU by the number of images processed. This provides an average IoU score across all images, giving us an overall measure of the performance of the image processing pipeline.<br>

### 2. Output:
The evaluation code displays the following information:
 * IoU for each image
 * The total IoU after each image 
 * mIoU for all the images
 
 # Overview 
The pipeline consists of the following steps:
1. **Reading Images** : Images are first read from each folder in the dataset.
2. **Creating an image processing pipeline folder** : A pipeline folder is created within the dataset folder to store the processing pipeline. 
3. **Creating an output folder** : An output folder is created within the dataset folder to store the final image.
4. **Converting to HSV** : Input images are converted from RGB to HSV colour space.
5. **Extracts the value channel** : Extracts the value channel (brightness) from the HSV image
6. **Preprocess the image** : Preprocessing techniques such as median filtering and Gaussian blurring are used to reduce noise.
7. **Thresholding/Segmentation** : Images are thresholded to seperate the objects from the background.
8. **Binary Image Processing** : Morphological operations like opening and closing are performed to further refine the segmentation.
9. **Additional Processing** : Perform further closing to fill small holes.
10. **Save the processing pipeline images** : The processing pipeline images (these include the original image, hsv image, v channel image, median filtered image, threshold image, cleaned image, output image) are saved in the image processing pipeline folder.
11. **Save the final image** : The final image is saved in the output folder.
12. **User input** : User is prompted to enter if they want any images displayed. If yes, the following steps are followed else the program ends.
13. **User selection** : The user enters the type (easy, medium, hard) and the index (1,2,3) of image they want displayed. 
14. **Display the images** : Depending on the type and index entered, the original image from the input_images file and the final image from the output_images file are displayed on the screen.


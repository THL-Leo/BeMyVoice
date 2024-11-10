import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Constants for image processing
offset = 20
imgSize = 300
counter = 0
max_images = 200  # Limit of images to capture

# Folder to save images
folder = "/Users/reetvikchatterjee/Desktop/Dataset/Work"

# Create directory if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Directory created at: {folder}")

while True:
    # Read frame from video capture
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Find hands in the image
    hands, img = detector.findHands(img)

    if len(hands) > 0:  # If at least one hand is detected
        # Get bounding boxes for detected hands
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

        # Expand the bounding box to ensure both hands are included with an offset
        x_min = max(0, x_min - offset)
        y_min = max(0, y_min - offset)
        x_max = min(img.shape[1], x_max + offset)
        y_max = min(img.shape[0], y_max + offset)

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop and resize hand region
        imgCrop = img[y_min:y_max, x_min:x_max]
        if imgCrop.size == 0:
            print("Empty image detected. Skipping...")
            continue

        # Resize hand image to match model input size
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        # Display cropped and resized hand image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgResize)

        # Automatically save image without needing "s" key
        if counter < max_images:
            counter += 1
            file_path = f'{folder}/Image_{counter}_{time.time()}.jpg'
            cv2.imwrite(file_path, imgResize)
            print(f"Image saved: {file_path}")

        # Check if max image limit reached
        if counter >= max_images:
            print(f"Reached the limit of {max_images} images. Exiting...")
            break

    # Display original image
    cv2.imshow('Image', img)

    # Check for key press
    if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

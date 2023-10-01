import os
from PIL import Image
import numpy as np
import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract and save face region from an image
def extract_face_and_save(image_path, output_directory):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Assuming the first detected face is the main face
        face_roi = img[y:y+h, x:x+w]
        # Resize the face region to a fixed size (e.g., 100x100 pixels)
        resized_face = cv2.resize(face_roi, (100, 100))
        # Extract the expression information from the input file name
        expression = os.path.splitext(os.path.basename(image_path))[0]
        # Save the resized face image with expression information and a counter as .png
        counter = 1
        output_filename = f"resized_{expression}_{counter}.png"
        while os.path.exists(os.path.join(output_directory, output_filename)):
            counter += 1
            output_filename = f"resized_{expression}_{counter}.png"
        cv2.imwrite(os.path.join(output_directory, output_filename), resized_face)
        print(f"Face extracted and saved successfully: {output_filename}")
    else:
        print(f"No face detected in the input image: {image_path}")

# Directory containing the input images
input_directory = 'C:/Users/admin/Desktop/Facial-Recognition-System/data/yalefaces'  # Replace with your dataset directory

# Output directory to save cropped and resized face images
output_directory = 'C:/Users/admin/Desktop/Facial-Recognition-System/data/yalefaces_resized'  # Replace with the desired output directory

# Ensure the output directory exists, if not, create it
os.makedirs(output_directory, exist_ok=True)

# Process all images in the input directory
for file_name in os.listdir(input_directory):
    # Check if the file is an image (you might need to adjust the valid file extensions)
    if file_name.lower().endswith(('.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink')):
        input_image_path = os.path.join(input_directory, file_name)
        extract_face_and_save(input_image_path, output_directory)

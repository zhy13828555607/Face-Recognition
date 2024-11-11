"""
Recognize faces from the original data, making it to the train data
"""

import cv2
import os


# Classification
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(image_path, output_dir):
    image = cv2.imread(image_path)  # Load picture
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gray
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]  # Get the area of human face
        face_filename = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i + 1}.jpg")
        cv2.imwrite(face_filename, face)  # Save

def process_dataset(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # file format
            image_path = os.path.join(input_dir, filename)
            print(f"Processing image: {image_path}")
            extract_faces(image_path, output_dir)

# Input and Output position
input_directory = r"E:/pythonproject/pythonProject/pythonproject/Face/original_images/mark_zuckerberg"
output_directory = "E:/pythonproject/pythonProject/pythonproject/Face/train_dataset/mark_zuckerberg"
process_dataset(input_directory, output_directory)
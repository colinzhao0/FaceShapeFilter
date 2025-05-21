import cv2
import dlib
import numpy as np
import math 
from pathlib import Path

from automatic import get_face_shape, distance, image_to_shape

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")



# Iterate through image files to test
folder_path = Path('faces/round')

total = 0
count = 0

for img_path in folder_path.glob('*'):
    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        try:
            print(f"Processing: {img_path.name}")
            shape = image_to_shape(str(img_path))
            if shape == "round" or shape == "oval":
                count+=1
            total += 1
            
            print(shape)
            
        except Exception as e:
            print(f"Error opening {img_path.name}: {e}")

print(f"{count} out of {total} correct")
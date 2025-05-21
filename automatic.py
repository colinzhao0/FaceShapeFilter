import cv2
import dlib
import numpy as np
import math 
from pathlib import Path

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_face_shape(landmarks):
    cheek_width = distance(landmarks[1], landmarks[15])
    face_height = distance(landmarks[8], landmarks[27])
    jaw_width = distance(landmarks[4], landmarks[12])
    forehead_width = distance(landmarks[17], landmarks[26])

    ratio_width_height = cheek_width / face_height

    if ratio_width_height > 0.9:
        return "round"
    elif ratio_width_height < 0.7:
        return "oblong"
    elif jaw_width / cheek_width > 0.9:
        return "square"
    elif forehead_width > cheek_width * 0.95 and jaw_width < cheek_width * 0.85:
        return "heart"
    else:
        return "oval"

#Read Image
def image_to_shape(path):
    img = cv2.imread(path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #get faces
    faces = detector(imgGray)

    #find landmarks
    for face in faces:
        landmarks = predictor(imgGray, face)
        landmarks_list = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y 
            cv2.circle(img, (x,y), 2, (0,255,0), -1)
            cv2.putText(img, str(n), (x,y), 1, .5, (0,255,0),1)
            landmarks_list.append((x,y))
        
        return(get_face_shape(landmarks_list))

# Iterate through image files to test
folder_path = Path('faces/round')

total = 0
count = 0

for img_path in folder_path.glob('*'):
    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        try:
            print(f"Processing: {img_path.name}")
            shape = image_to_shape(str(img_path))
            if shape == "round":
                count+=1
            total += 1
            
            print(shape)
            
        except Exception as e:
            print(f"Error opening {img_path.name}: {e}")

print(f"{count} out of {total} correct")
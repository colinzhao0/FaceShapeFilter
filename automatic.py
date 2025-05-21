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
    # Key landmarks indices (dlib's 68-point model)
    LEFT_CHEEK = 1
    RIGHT_CHEEK = 15
    CHIN = 8
    FOREHEAD = 27
    JAW_LEFT = 4
    JAW_RIGHT = 12
    LEFT_TEMPLE = 17
    RIGHT_TEMPLE = 26

    # Calculate widths and heights
    cheek_width = distance(landmarks[LEFT_CHEEK], landmarks[RIGHT_CHEEK])
    face_height = distance(landmarks[CHIN], landmarks[FOREHEAD])
    jaw_width = distance(landmarks[JAW_LEFT], landmarks[JAW_RIGHT])
    forehead_width = distance(landmarks[LEFT_TEMPLE], landmarks[RIGHT_TEMPLE])

    # Ratios
    ratio_width_height = cheek_width / face_height
    ratio_jaw_cheek = jaw_width / cheek_width
    ratio_forehead_cheek = forehead_width / cheek_width

    # Shape determination (prioritize most distinct features first)
    # 1. Heart-shaped: Narrow jaw + wide forehead
    if ratio_forehead_cheek > 0.95 and ratio_jaw_cheek < 0.85:
        return "heart"
    
    # 2. Square: Jaw width ≈ cheek width + angular jaw (check chin landmarks)
    print(ratio_jaw_cheek)
    if ratio_jaw_cheek > 0.86:  # Almost straight line
        return "square"
    
    # 3. Round: Width ≈ Height
    if 0.85 <= ratio_width_height <= 1.1:
        return "round"
    
    # 4. Oblong: Height >> Width
    if ratio_width_height < 0.75:
        return "oblong"
    
    # 5. Default to oval
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

#Run functions on image
path = 'resources/face.jpg' 
print(image_to_shape(path))
import cv2
import numpy as np

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a window and trackbars
cv2.namedWindow('Face Shape Detector')
cv2.createTrackbar('Shape', 'Face Shape Detector', 0, 6, lambda x: None)
cv2.createTrackbar('Opacity', 'Face Shape Detector', 50, 100, lambda x: None)
cv2.createTrackbar('Thickness', 'Face Shape Detector', 2, 10, lambda x: None)
cv2.createTrackbar('Width', 'Face Shape Detector', 0, 150, lambda x: None)

# Shape definitions

def get_center_x(x_pos, height):
    return x_pos + int(height / 2)

def get_heart_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    cy = y + h // 2
    shift = int(h * 0.1)
    points = []
    # Left lobe
    for i in range(135, 226, 5):
        angle = np.radians(i)
        px = cx - w // 4 + int((w // 4) * np.cos(angle))
        py = cy - h // 6 + int((h // 4) * np.sin(angle)) + shift
        points.append((px, py))
    # Right lobe
    for i in range(-45, 46, 5):
        angle = np.radians(i)
        px = cx + w // 4 + int((w // 4) * np.cos(angle))
        py = cy - h // 6 + int((h // 4) * np.sin(angle)) + shift
        points.append((px, py))
    # Bottom tip
    points.append((cx, y + h + shift))
    return np.array(points, dtype=np.int32)

def get_square_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    return np.array([
        (cx - int(w / 2), y),
        (cx + int(w / 2), y),
        (cx + int(w / 2), y + h),
        (cx - int(w / 2), y + h)
    ], dtype=np.int32)

def get_round_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    return cv2.ellipse2Poly((cx, y + h // 2), (w // 2, h // 2), 0, 0, 360, 10)

def get_oblong_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    return np.array([
        (cx - w // 4, y+h),
        (cx + w // 4, y+h),
        (cx + w // 2, y+int(h*.1)),
        (cx - w // 2, y+int(h*.1))
    ], dtype=np.int32)

def get_triangle_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    return np.array([
        (cx, y+int(h*1.2)),
        (cx + w // 2, y+int(h*.2)),
        (cx - w // 2, y+int(h*.2))
    ], dtype=np.int32)

def get_kite_shape(face_rect):
    x, y, w, h = face_rect
    cx = get_center_x(x, h)
    return np.array([
        (cx, y),
        (cx + w // 2, y + h // 1.8),
        (cx, y + int(h*1.2)),
        (cx - w // 2, y + h // 1.8)
    ], dtype=np.int32)

shape_functions = [
    get_heart_shape, get_square_shape,
    get_round_shape, get_oblong_shape, get_triangle_shape, get_kite_shape
]

shape_names = [
    "Heart", "Square/Rectangle", "Round", 
    "Oblong", "Triangle/Pear", "Kite"
]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create a copy for overlay
    overlay = frame.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Get current trackbar positions
    shape_idx = cv2.getTrackbarPos('Shape', 'Face Shape Detector')
    opacity = cv2.getTrackbarPos('Opacity', 'Face Shape Detector') / 100
    thickness = cv2.getTrackbarPos('Thickness', 'Face Shape Detector')
    width = cv2.getTrackbarPos('Width', 'Face Shape Detector')
    
    # Process each detected face
    for (x, y, w, h) in faces:

        # Adjust width and height based on the trackbar
        w = int(w * (width / 100))

        # Get the selected shape
        shape_points = shape_functions[shape_idx]((x, y, w, h))
        print(w, h)
        
        # Draw just the outline of the shape (not filled)
        cv2.polylines(overlay, [shape_points], isClosed=True, 
                     color=(0, 255, 255), thickness=thickness)
    
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    
    # Display the current shape name
    cv2.putText(frame, f"Shape: {shape_names[shape_idx]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Thickness: {thickness}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the result
    cv2.imshow('Face Shape Detector', frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
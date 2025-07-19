import cv2
import mediapipe as mp
import numpy as np
import math
from FaceDetectionModule import FaceDetector as fd
import joblib
import pandas as pd  

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

#-------------------------------------------------------------------------------------------------
#Factory Functions
def extract_features(landmarks, width, height):
    ear = EAR(landmarks, width, height)
    mar = MAR(landmarks, width, height)
    pitch_val = pitch(landmarks, width, height)
    roll_val = roll(landmarks, width, height)
    return [ear, mar, pitch_val, roll_val]

def distance(p1,p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def EAR(lms,width,height):
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [263, 387, 385, 362, 380, 373]
    #Left EAR CALC
    points = []
    for idx in left_eye_indices:
        lm = lms[idx]
        points.append((lm.x * width, lm.y * height))
    # P1, P2, P3, P4, P5, P6
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    left_ear = (vertical1 + vertical2) / (2.0 * horizontal)
    #Right EAR CALC
    points = []
    for idx in right_eye_indices:
        lm = lms[idx]
        points.append((lm.x * width, lm.y * height))
    # P1, P2, P3, P4, P5, P6
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    right_ear = (vertical1 + vertical2) / (2.0 * horizontal)
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear

def MAR(lms,width,height):
    #Mouth landmarks
    left = lms[61]
    right = lms[291]
    top = lms[13]
    bottom = lms[14]
    #Convert to pixel coordinates
    left = (left.x * width, left.y * height)
    right = (right.x * width, right.y * height)
    top = (top.x * width, top.y * height)
    bottom = (bottom.x * width, bottom.y * height)
    #Calculate MAR (Mouth aspect ratio)
    horizontal = distance(left,right)
    vertical = distance(top,bottom)
    mar = vertical / horizontal
    return mar

def pitch(lms, width, height):
    # 3D model points (assumed standard face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -63.6, -12.5),     # Chin
        (-43.3, 32.7, -26.0),    # Left eye corner
        (43.3, 32.7, -26.0),     # Right eye corner
        (-28.9, -28.9, -24.1),   # Left mouth corner
        (28.9, -28.9, -24.1)     # Right mouth corner
    ])
    image_points = np.array([
        (lms[1].x * width, lms[1].y * height),   # Nose tip
        (lms[152].x * width, lms[152].y * height), # Chin
        (lms[33].x * width, lms[33].y * height),   # Left eye
        (lms[263].x * width, lms[263].y * height), # Right eye
        (lms[61].x * width, lms[61].y * height),   # Left mouth
        (lms[291].x * width, lms[291].y * height)  # Right mouth
    ], dtype="double")

    # Camera parameters (assume no distortion)
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
    pitch = math.atan2(-rotation_matrix[2,0], sy) * (180 / math.pi) 
    return pitch
def roll(lms, width, height):
    forehead = lms[10]
    chin = lms[152]
    x1, y1 = forehead.x * width, forehead.y * height
    x2, y2 = chin.x * width, chin.y * height
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_rad = np.arctan2(delta_x, delta_y)  # X and Y are swapped to get roll
    angle_deg = np.degrees(angle_rad)
    return angle_deg
def aggregate_features(window):
    window = np.array(window)
    return [
        np.mean(window[:, 0]), np.std(window[:, 0]),
        np.mean(window[:, 1]), np.std(window[:, 1]),
        np.mean(window[:, 2]), np.std(window[:, 2]),
        np.mean(window[:, 3]), np.std(window[:, 3])
    ]
#----------------------------------------------------------------------------------------------------------
win_size = 20
#stride = 5
detector = fd()
cap = cv2.VideoCapture(1)
window = []
model = joblib.load(r'Saved_Models\drowsiness_model.pkl') 
result = 0
#Feature names for consistent prediction
feature_names = [
    'mean_ear', 'std_ear',
    'mean_mar', 'std_mar',
    'mean_pitch', 'std_pitch',
    'mean_roll', 'std_roll'
]
#--------------------------------------------------------------------------------------------------------------
while True:
    success, img = cap.read()
    height, width = img.shape[:2]
    if not success:
        print("Failed to capture image")
        break
    lms = detector.detect(img)
    if lms:
        mp_drawing.draw_landmarks(
        img,
        lms,
        mp_face_mesh.FACEMESH_TESSELATION,   # Full face mesh
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))
        #------------------------------------------------------------------------------------------------------
        features = extract_features(lms.landmark,width,height)
        window.append(features)
        if len(window)>win_size:
            window.pop(0)
        if len(window) == win_size:
            data = aggregate_features(window)
            # Wrap data into DataFrame to preserve feature names
            df = pd.DataFrame([data], columns=feature_names)
            result = model.predict(df)[0]
            result = 'Drowsy' if result == 1 else "Alert"
        cv2.putText(img, f'Label: {result}', (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        #---------------------------------------------------------------------------------------------------------
    cv2.imshow("Webcam Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

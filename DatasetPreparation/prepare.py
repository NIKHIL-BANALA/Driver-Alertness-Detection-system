import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import mediapipe as mp
import numpy as np
import math
from FaceDetectionModule import FaceDetector as fd
import csv


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
detector = fd()
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


#----------------------------------------------------------------------------------------------------------

def write_data(window, label, csv_file):
    # Convert to NumPy array for easy slicing
    window = np.array(window)  # shape: (window_size, 4)
    ear_list = window[:, 0]
    mar_list = window[:, 1]
    pitch_list = window[:, 2]
    roll_list = window[:, 3]
    # Aggregate features
    features = {
        'mean_ear': np.mean(ear_list),
        'std_ear': np.std(ear_list),
        'mean_mar': np.mean(mar_list),
        'std_mar': np.std(mar_list),
        'mean_pitch': np.mean(pitch_list),
        'std_pitch': np.std(pitch_list),
        'mean_roll': np.mean(roll_list),
        'std_roll': np.std(roll_list),
        'label': label
    }
    # Write to CSV
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=features.keys())
        writer.writerow(features)

def process_batch(folder_path, file_path,label, start_idx, batch_size=5000, win_size=20, stride=5):
    filenames = sorted(os.listdir(folder_path))
    selected_files = filenames[start_idx:start_idx + batch_size]
    count = 0
    window = []
    for filename in selected_files:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            height, width = img.shape[:2]
            lms = detector.detect(img)
            if lms:
                features = extract_features(lms.landmark, width, height)
                window.append(features)
                if len(window) == win_size:
                    write_data(window, label, file_path)
                    window = window[stride:]
                    count+=1
                    sys.stdout.write(f"\rinserted {count} rows")
                    sys.stdout.flush()

#----------------------------------------------------------------------------------------------
drowsy_folder = r"\Path_Here"
drowsy_csv = r"\Path_Here"
not_drowsy_folder = r"\Path_Here"
not_drowsy_csv = r"\Path_Here"
#----------------------------------------------------------------------------------------------------
#Model Trained on
#drowsy-25000
#process_batch(drowsy_folder,drowsy_csv,1,0,batch_size = 5000)
#process_batch(drowsy_folder,drowsy_csv,1,start_idx = 5000,batch_size = 10000)
#process_batch(drowsy_folder,drowsy_csv,1,start_idx = 15000,batch_size = 10000)

#not_drowsy-25000
#process_batch(not_drowsy_folder,not_drowsy_csv,0,0,batch_size = 5000)
#process_batch(not_drowsy_folder,not_drowsy_csv,0,start_idx=5000,batch_size = 5000)
#process_batch(not_drowsy_folder,not_drowsy_csv,0,start_idx=10000,batch_size = 5000)
#process_batch(not_drowsy_folder,not_drowsy_csv,0,start_idx=15000,batch_size = 10000)
#-------------------------------------------------------------------------------------------------------------
print("started")
#Place batch specific line here
print("completed")

import mediapipe as mp
import cv2
import numpy as np
"""
multi_face_landmarks = [
    # Face 1
    NormalizedLandmarkList(landmark=[
        NormalizedLandmark(x=0.3, y=0.4, z=-0.02),  # landmark 0
        NormalizedLandmark(x=0.5, y=0.6, z=-0.03),  # landmark 1
    ]),
    
    # Face 2
    NormalizedLandmarkList(landmark=[
        NormalizedLandmark(x=0.2, y=0.3, z=-0.01),  # landmark 0
        NormalizedLandmark(x=0.4, y=0.5, z=-0.02),  # landmark 1
    ])
]


"""


class FaceDetector:
    def __init__(self,max_faces = 1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # For iris detection
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
    def detect(self,img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(img_rgb)#will be multi face landmarks
        if not result.multi_face_landmarks:
            return None
        closest_face_lms = None
        depth = float('inf')
        for face_landmarks in result.multi_face_landmarks:
            if face_landmarks is not None:
                # Calculate the average depth of the face landmarks
                key_ids = [1, 4, 33, 133, 263, 362, 61, 291, 199, 152]
                avg_depth = np.average([face_landmarks.landmark[i].z for i in key_ids])
                if avg_depth < depth:
                    depth = avg_depth
                    closest_face_lms = face_landmarks
        return closest_face_lms

            
        


## 💡 Intuition Behind the Project

Drowsy driving is a leading cause of road accidents. This project provides a solution by analyzing the driver’s **facial landmarks** in real-time and classifying the alertness level. The system focuses on **eye aspect ratio (EAR)**, **mouth aspect ratio (MAR)**, and **head posture (pitch & roll)** to determine drowsiness. The data is processed over a time window to handle real-world noise and variability. The model predicts driver state using temporal features aggregated over 20 frames.

## 🔧 Features Considered

| Feature | Description |
|----------|-------------|
| **EAR**  | Measures eye closure by comparing eye landmark distances |
| **MAR**  | Measures mouth openness to detect yawning |
| **Pitch** | Detects head nodding or drooping forward |
| **Roll**  | Detects side tilting of the head |

Each feature is statistically processed using:

- **Mean** over window of 20 frames  
- **Standard Deviation** over window  

Final feature vector per sample: **8 features**.

## 🗄️ Data Preparation Pipeline

- Images from [NTHU DDD Dataset (Kaggle)](https://www.kaggle.com/datasets/banudeep/nthuddd2) were used.
- For each image, **Mediapipe FaceMesh** extracts facial landmarks.
- Using sliding windows (size = 20, stride = 5), statistical features were computed:
    - `mean` and `std` of EAR, MAR, Pitch, Roll
- Generated two CSV files:  
    - **drowsy.csv** (label = 1)  
    - **not_drowsy.csv** (label = 0)  
- Final CSV was balanced to avoid bias.
- The **prepare.py** script automates this process.

## 🧠 Model Training Details

- **Algorithm Used:** Random Forest Classifier  
- **Reason:** Handles non-linear data, robust to outliers, low tuning required.
- **Training Pipeline:**
    - Combined `drowsy.csv` and `not_drowsy.csv`
    - Randomized and split into train/test
    - Model trained on **50,000 samples** (25k drowsy + 25k not_drowsy)
- **Optimization:**
    - Feature scaling where needed (especially for pitch/roll)
    - Handled inconsistent pitch variations via data smoothing during preparation, but final model is robust even without smoothing
- **Output Model:** `Saved_Models/drowsiness_model.pkl`

## 🚀 Real-Time Detection

- Webcam feed is captured using OpenCV.
- **Mediapipe FaceMesh** extracts live landmarks.
- A **sliding window of 20 frames** collects features.
- The trained Random Forest model classifies the driver’s state as **Alert** or **Drowsy**.
- Output displayed on the screen overlay.

## 📦 Modules & Installation

Install all required modules using:

```bash
pip install mediapipe opencv-python scikit-learn numpy pandas joblib

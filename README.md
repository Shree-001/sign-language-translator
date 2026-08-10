<<<<<<< HEAD
# sign-language-translator
=======
# Real-Time Static ASL Sign Language Translator

A lightweight, real-time static American Sign Language (ASL) translator built with **MediaPipe Hands**, **Scikit-Learn (RandomForestClassifier)**, **OpenCV**, and **pyttsx3**.

## Features

- **MediaPipe Hand Landmark Tracking**: Extracts 21 3D spatial joint coordinates per hand (\(63\) features) in real time.
- **Hand-Relative Scale & Translation Normalization**: Translates wrist to origin and scales by maximum Euclidean distance for distance/framing invariance.
- **Handedness Mirroring**: Automatically projects right hand keypoints into left hand coordinate space to match training dataset orientation.
- **RandomForest Classifier**: Trained on extracted landmarks with **99.52% evaluation accuracy**.
- **Offline Text-To-Speech**: Real-time audio output with non-blocking daemon threading and a 2.0s cooldown.

## Project Structure

```
sign-language-translator/
├── data/
│   └── landmarks.csv           # Normalized landmark feature dataset
├── models/
│   └── asl_classifier.pkl      # Trained Random Forest model
├── src/
│   ├── extract_landmarks.py    # Extracts normalized keypoints from training dataset
│   ├── train_classifier.py     # Trains RandomForestClassifier & evaluates metrics
│   └── realtime_predict.py     # Live OpenCV camera inference & TTS speech
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd sign-language-translator

# Activate virtual environment
venv\Scripts\activate

# Install requirements
pip install opencv-python "mediapipe<0.10.30" numpy pandas scikit-learn pyttsx3 joblib
```

### 2. Run Real-Time Translator

```bash
python src/realtime_predict.py
```

Press **`q`** to exit the camera window.
>>>>>>> cc2fb8d (Initial commit: Real-time static ASL sign language translator)

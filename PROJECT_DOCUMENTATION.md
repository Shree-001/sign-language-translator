# Real-Time Static ASL Sign Language Translator
## Complete Technical Overview, Workflow, & Defense Guide

---

## 1. Executive Summary

The **Real-Time Static ASL Sign Language Translator** is an intelligent computer vision and machine learning system that translates American Sign Language (ASL) hand gestures into text and spoken audio in real time using standard webcam hardware.

Unlike traditional deep learning approaches that rely on heavy Convolutional Neural Networks (CNNs) processing millions of raw image pixels, this project leverages **MediaPipe Hands** for 3D geometric landmark extraction, combined with a **Random Forest Classifier**. This architecture enables ultra-fast, highly accurate (**99.52% accuracy**), and CPU-friendly real-time sign recognition.

---

## 2. Core Problem & Solution Approach

### The Problem
Deaf and hard-of-hearing individuals often face communication barriers when interacting with people who do not know Sign Language. Existing computer vision solutions are often:
- **Resource Intensive**: Require powerful GPUs to run deep neural networks.
- **Environment Sensitive**: Struggle when background lighting, skin tones, or camera distances change.
- **Sluggish**: High latency causes video lag and stuttering during text-to-speech rendering.

### Our Solution
1. **Feature Extraction over Raw Pixels**: We extract **21 3D spatial keypoints** (63 numeric coordinates) representing hand joints rather than passing 120,000+ raw image pixels into a model.
2. **Mathematical Normalization**: We apply wrist-origin translation and scale normalization to make recognition invariant to distance, hand size, and positioning.
3. **Handedness Alignment**: Automatically mirrors right-hand keypoints into left-hand coordinate space to maintain full consistency regardless of which hand the user uses.
4. **Lightweight Classification**: Uses **Random Forest Classification** (200 decision trees) for near-instantaneous inference on ordinary laptop CPUs.
5. **Non-Blocking Speech Synthesis**: Runs offline Text-to-Speech (`pyttsx3`) inside a background thread with a 2-second cooldown to prevent audio overlapping or frame drops.

---

## 3. System Architecture & Workflow

```
+-------------------+      +-----------------------+      +-------------------------+
|  Training Images  | ---> |   MediaPipe Hands     | ---> |  Hand-Relative Scaling  |
|   (Kaggle ASL)    |      | (21 Keypoints x,y,z)  |      |   & Wrist Translation   |
+-------------------+      +-----------------------+      +-------------------------+
                                                                       |
                                                                       v
+-------------------+      +-----------------------+      +-------------------------+
| Saved Model File  | <--- | Random Forest Trainer | <--- |   Feature Dataset CSV   |
| (asl_classifier)  |      | (200 Decision Trees)  |      |   (data/landmarks.csv)  |
+-------------------+      +-----------------------+      +-------------------------+
          |
          v
+-------------------+      +-----------------------+      +-------------------------+
| Live Webcam Feed  | ---> | Real-Time Predictions | ---> | Text Banner & Offline   |
| (OpenCV Capture)  |      | & Visual Skeleton     |      | Speech (pyttsx3)        |
+-------------------+      +-----------------------+      +-------------------------+
```

### Stage 1: Landmark Extraction (`src/extract_landmarks.py`)
- Reads images from 29 ASL classes (`A–Z`, `space`, `del`).
- Uses `MediaPipe Hands` to locate 21 key points on the hand.
- Applies **Wrist-Origin Normalization** and **Max-Distance Scaling**.
- Exports 63 features + 1 label column to `data/landmarks.csv` (7,348 rows).

### Stage 2: Classifier Training (`src/train_classifier.py`)
- Splits data into 80% training and 20% testing sets using stratified sampling.
- Trains a `RandomForestClassifier` with 200 decision trees.
- Evaluates test accuracy (**99.52%**) and exports `models/asl_classifier.pkl`.

### Stage 3: Real-Time Inference & Speech (`src/realtime_predict.py`)
- Captures live camera frames using OpenCV.
- Processes un-flipped frames through MediaPipe Hands.
- Checks handedness (`results.multi_handedness`): if a Right hand is shown, mirrors x-coordinates (`-x`) into the training space.
- Normalizes landmarks, feeds them into `asl_classifier.pkl`, and renders the visual skeleton.
- Triggers `pyttsx3` text-to-speech in a separate daemon thread respecting a 2.0-second cooldown.

---

## 4. Key Engineering & Technical Solutions

| Technical Challenge | Root Cause | Engineering Solution |
| :--- | :--- | :--- |
| **Framing & Distance Invariance** | Raw MediaPipe outputs range 0 to 1 relative to the full image frame, making predictions dependent on camera distance. | **Wrist Translation + Scale Normalization**: Subtract wrist coordinate \((x_0, y_0, z_0)\) from all landmarks and divide by max Euclidean distance. |
| **Left vs Right Hand Disparity** | Training dataset consisted of Left-hand images, causing Right-hand gestures to fail. | **Handedness Inversion**: Detect hand label via `multi_handedness`. Invert x-coordinates (`-x`) for Right hands before normalization. |
| **Video Window Lag during Speech** | `pyttsx3.runAndWait()` is synchronous and blocks the main thread. | **Daemon Threading**: Speech engine calls run asynchronously in a background thread without blocking the OpenCV render loop. |
| **Audio Repeating Spam** | Live predictions execute every frame (30 FPS), triggering continuous audio loops. | **Cooldown Timer**: Enforce a 2-second delay between spoken words unless the predicted sign changes. |

---

## 5. Real-World Applications & Uses

1. **Assistive Communication**: Helps non-signers communicate effortlessly with deaf individuals in daily life, retail, and public services.
2. **Educational ASL Learning Tool**: Provides real-time visual feedback and audio confirmation for students learning ASL gestures.
3. **Accessibility Kiosks**: Can be integrated into public service desks, airports, hospitals, or customer service terminals.
4. **Embedded & Edge Robotics**: Extremely low CPU footprint makes it suitable for deployment on low-cost devices like Raspberry Pi or jetson boards.

---

## 6. How to Run the Project

### Prerequisites
- Python 3.10 / 3.11 / 3.12
- Webcam connected to computer

### Step-by-Step Setup

1. **Navigate to project directory**:
   ```bash
   cd c:\Users\shrin\sign-language-translator
   ```

2. **Activate virtual environment**:
   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Real-Time Translator**:
   ```bash
   python src/realtime_predict.py
   ```
   *Press **`q`** on the video window to quit.*

---

## 7. Viva Defense & Interview Cheat Sheet

#### Q1: Why did you use MediaPipe landmarks instead of raw pixel images with a CNN?
> **Answer**: Raw images contain over 120,000 pixels filled with background noise, varying lighting, skin tones, and camera artifacts. MediaPipe isolates the hand geometry into 21 key structural points (63 floats). This reduces feature dimensionality by 99.9%, drastically lowering compute requirements and enabling real-time CPU performance.

#### Q2: Why Random Forest instead of a Deep Learning Neural Network?
> **Answer**: Once MediaPipe performs non-linear visual spatial extraction, the data becomes structured tabular coordinates. Random Forest creates an ensemble of 200 decision trees that excels at non-linear boundary classification on low-dimensional tabular data, reaching 99.52% accuracy in seconds without needing GPUs.

#### Q3: How did you ensure the model works when the hand is closer or farther from the camera?
> **Answer**: We implemented custom hand-relative normalization. First, we translate the wrist landmark to origin \((0,0,0)\). Then, we divide all coordinates by the maximum 3D Euclidean distance between the wrist and any joint. This makes the feature vector completely independent of hand size or camera distance.

#### Q4: How does your system handle Left vs. Right hand gestures?
> **Answer**: The training dataset predominantly contained left-hand gesture images. During live webcam inference, our code inspects `multi_handedness`. If a Right hand is detected, we mirror its x-coordinates (`x = -x`) prior to normalization. This seamlessly maps Right-hand gestures into the model's trained Left-hand feature space.

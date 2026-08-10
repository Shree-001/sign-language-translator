"""
src/realtime_predict.py

VIVA EXPLANATION — REAL-TIME INFERENCE & SPEECH COOLDOWN DESIGN
--------------------------------------------------------------
1. MediaPipe Real-Time Tracking: Processes live video stream frame-by-frame on CPU, extracting
   normalized 21 hand spatial joints without heavy deep learning overhead.
2. Non-Blocking TTS Threading: pyttsx3.runAndWait() is synchronous and blocking. Running speech
   in a daemon thread ensures the OpenCV video rendering loop never lags or drops frames.
3. Cooldown Mechanism (~2.0s): Prevents continuous audio spamming while the user holds a static sign.
"""

import os
import time
import threading
import cv2
import joblib
import numpy as np
import pyttsx3
import mediapipe as mp

MODEL_PATH = os.path.join("models", "asl_classifier.pkl")
COOLDOWN_SECONDS = 2.0

# Initialize Text-To-Speech engine in a non-blocking thread to avoid freezing video loop
last_spoken_time = 0
last_spoken_label = None

def speak_text(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

def normalize_landmarks(landmark_list):
    # landmark_list: list of 21 (x, y, z) tuples from MediaPipe
    base_x, base_y, base_z = landmark_list[0]  # wrist as the origin
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmark_list]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    normalized = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]
    return normalized

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}. Run train_classifier.py first.")

    model = joblib.load(MODEL_PATH)
    feature_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    import pandas as pd

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = None
    for cam_idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                print(f"Webcam initialized successfully on camera index {cam_idx}. Press 'q' to exit.")
                break
            temp_cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError("No active webcam device found. Please check your camera connection.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Process raw un-flipped frame for detection matching training orientation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        predicted_label = "No hand"

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check handedness (mirror Right hand to Left hand coordinate space matching training dataset)
                handedness_label = "Left"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness_label = results.multi_handedness[idx].classification[0].label

                if handedness_label == "Right":
                    raw_landmarks = [(-lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                else:
                    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                norm_landmarks = normalize_landmarks(raw_landmarks)
                features = []
                for x, y, z in norm_landmarks:
                    features.extend([x, y, z])

                features_df = pd.DataFrame([features], columns=feature_cols)
                predicted_label = model.predict(features_df)[0]

                # Trigger offline TTS audio output if cooldown period has elapsed
                current_time = time.time()
                if (current_time - last_spoken_time > COOLDOWN_SECONDS) or (predicted_label != last_spoken_label):
                    speak_text(predicted_label)
                    last_spoken_time = current_time
                    last_spoken_label = predicted_label

        # Overlay text prediction banner on video frame
        cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {predicted_label}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Flip display frame horizontally for mirror UX view
        display_frame = cv2.flip(frame, 1)
        cv2.imshow("ASL Real-Time Translator", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

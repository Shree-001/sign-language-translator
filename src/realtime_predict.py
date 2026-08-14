"""
src/realtime_predict.py

VIVA EXPLANATION — REAL-TIME INFERENCE, MAJORITY-VOTE STABILITY & MULTI-DIGIT STATE MACHINE
-----------------------------------------------------------------------------------------
1. Dual Model Architecture: Letter classifier (asl_classifier.pkl) and Digit classifier
   (digit_classifier.pkl) are segregated to eliminate misclassification between identical shapes
   (e.g., 1 vs D, 2 vs V, 0 vs O). Mode toggling ('l' / 'n') switches active classifier.
2. 10-Frame Majority-Vote Stability Filter: Maintains a rolling deque(maxlen=10) of raw per-frame
   predictions. A prediction is only confirmed when >= 7 out of 10 frames agree. This eliminates
   single-frame jitter and prevents over-triggering audio speech calls.
3. Hand-Absence Sequence Completion (2.0s): When no hand is in frame for 2 seconds, the multi-digit
   sequence is treated as complete, speaking the full number out loud and resetting the buffer.
4. Thread-Safe SAPI5 Worker: Persistent queue & single daemon thread with pythoncom.CoInitialize()
   and per-utterance pyttsx3 engine lifecycle to ensure reliable speech output on Windows.
"""

import os
import time
import queue
import threading
from collections import deque, Counter
import cv2
import joblib
import pandas as pd
import mediapipe as mp

LETTER_MODEL_PATH = os.path.join("models", "asl_classifier.pkl")
DIGIT_MODEL_PATH = os.path.join("models", "digit_classifier.pkl")

# Timers & Cooldowns
LETTER_COOLDOWN = 2.0          # Seconds between letter speech outputs
HAND_ABSENT_TIMEOUT = 2.0      # Seconds of no hand before auto-speaking completed number

# Thread-safe persistent TTS Queue & Worker Thread
speech_queue = queue.Queue()

def speech_worker():
    import pyttsx3
    import traceback
    import pythoncom
    pythoncom.CoInitialize()
    print("[TTS Worker] Thread started, COM initialized")
    try:
        while True:
            text = speech_queue.get()
            if text is None:
                break
            print(f"[TTS Worker] Speaking: {text}")
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            del engine
            speech_queue.task_done()
    except Exception as e:
        print(f"[TTS Worker Exception] {e}")
        traceback.print_exc()
    finally:
        pythoncom.CoUninitialize()
        print("[TTS Worker] Thread exiting, COM uninitialized")

# Start single daemon TTS worker thread at startup
tts_thread = threading.Thread(target=speech_worker, daemon=True)
tts_thread.start()

def speak_text(text):
    print(f"[TTS Call] Putting into queue: {text}")
    speech_queue.put(text)

# State Variables
last_spoken_time = 0
last_spoken_label = None

def normalize_landmarks(landmark_list):
    base_x, base_y, base_z = landmark_list[0]  # wrist as the origin
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmark_list]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    return [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]

if __name__ == "__main__":
    letter_model = joblib.load(LETTER_MODEL_PATH) if os.path.exists(LETTER_MODEL_PATH) else None
    digit_model = joblib.load(DIGIT_MODEL_PATH) if os.path.exists(DIGIT_MODEL_PATH) else None

    if letter_model is None and digit_model is None:
        raise FileNotFoundError("No trained model files found in models/. Run training scripts first.")

    current_mode = "LETTER" if letter_model is not None else "NUMBER"
    feature_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    # Rolling 10-frame majority-vote prediction buffer for noise filtering
    prediction_buffer = deque(maxlen=10)

    # Multi-digit sequencing state machine
    number_buffer = ""
    candidate_digit = None
    digit_confirmed = False
    last_hand_seen_time = time.time()

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
                print(f"Webcam initialized successfully on index {cam_idx}. Press 'l'/'n' for modes, 'c' to clear, 'q' to exit.")
                break
            temp_cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError("No active webcam device found. Please check your camera connection.")

    last_thread_check = time.time()

    while cap.isOpened():
        if time.time() - last_thread_check > 5.0:
            print(f"[TTS Status] Worker alive: {tts_thread.is_alive()}, Queue size: {speech_queue.qsize()}")
            last_thread_check = time.time()

        ret, detection_frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Pass raw unflipped detection_frame to MediaPipe
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        raw_label = "No hand"
        confirmed_label = "No hand"
        active_model = letter_model if current_mode == "LETTER" else digit_model

        if results.multi_hand_landmarks and active_model is not None:
            last_hand_seen_time = time.time()

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(detection_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                raw_label = str(active_model.predict(features_df)[0])
                prediction_buffer.append(raw_label)

                # Majority-vote filter: require >= 7 of last 10 frames to agree
                if len(prediction_buffer) == 10:
                    most_common, count = Counter(prediction_buffer).most_common(1)[0]
                    if count >= 7:
                        confirmed_label = most_common

                current_time = time.time()

                if current_mode == "LETTER":
                    # Speak confirmed letter only when stabilized
                    if confirmed_label != "No hand":
                        if confirmed_label != last_spoken_label:
                            speak_text(confirmed_label)
                            last_spoken_label = confirmed_label
                            last_spoken_time = current_time
                        elif (current_time - last_spoken_time > LETTER_COOLDOWN):
                            speak_text(confirmed_label)
                            last_spoken_time = current_time

                elif current_mode == "NUMBER":
                    # Digit sequencing using majority-vote confirmed label
                    if confirmed_label != "No hand":
                        if confirmed_label != candidate_digit:
                            candidate_digit = confirmed_label
                            digit_confirmed = False
                        else:
                            if not digit_confirmed:
                                number_buffer += candidate_digit
                                digit_confirmed = True

        else:
            # Clear rolling prediction buffer and candidate tracking on hand loss
            prediction_buffer.clear()
            candidate_digit = None
            digit_confirmed = False

            if current_mode == "NUMBER" and len(number_buffer) > 0:
                absent_duration = time.time() - last_hand_seen_time
                if absent_duration >= HAND_ABSENT_TIMEOUT:
                    print(f"Number Sequence Complete: {number_buffer}")
                    speak_text(number_buffer)
                    number_buffer = ""

        # Flip detection frame for mirror view AFTER skeleton is drawn
        display_frame = cv2.flip(detection_frame, 1)

        # Draw UI Overlay Banners on display_frame so text reads left-to-right (un-mirrored)
        mode_color = (0, 255, 0) if current_mode == "LETTER" else (255, 165, 0)
        cv2.rectangle(display_frame, (10, 10), (480, 90), (0, 0, 0), -1)
        cv2.putText(display_frame, f"MODE: {current_mode}  [n: Number | l: Letter]", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

        display_prediction = confirmed_label if confirmed_label != "No hand" else (f"{raw_label} (stabilizing...)" if raw_label != "No hand" else "No hand")

        if current_mode == "LETTER":
            cv2.putText(display_frame, f"Sign: {display_prediction}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, f"Sign: {display_prediction} | Number: {number_buffer}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ASL Real-Time Translator", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            current_mode = "LETTER"
            number_buffer = ""
            prediction_buffer.clear()
            print("Switched to LETTER Mode.")
        elif key == ord('n'):
            if digit_model is None:
                print("Digit model missing! Train digit_classifier.pkl first.")
            else:
                current_mode = "NUMBER"
                number_buffer = ""
                prediction_buffer.clear()
                print("Switched to NUMBER Mode.")
        elif key == ord('c'):
            number_buffer = ""
            prediction_buffer.clear()
            print("Cleared number buffer.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

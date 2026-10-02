"""
src/realtime_predict.py

BIDIRECTIONAL & MULTI-MODE ASL TRANSLATOR INFERENCE APP
-------------------------------------------------------
Modes:
1. LETTER Mode ('l'): Alphabet letters (A–Z, space, del) using asl_classifier.pkl.
2. NUMBER Mode ('n'): Numbers (0–9) with multi-digit state machine using digit_classifier.pkl.
3. DYNAMIC WORD Mode ('d'): Dynamic (word-level) signs using MediaPipe Holistic & PyTorch LSTM model.
4. PAUSED / IDLE Mode ('p'): Halts gesture recognition and audio speech to prevent false-positive
   classifications when the signer is resting hands, adjusting clothing, or scratching.
"""

import os
import time
import queue
import threading
from collections import deque, Counter
import cv2
import joblib
import pandas as pd
import numpy as np
import mediapipe as mp

LETTER_MODEL_PATH = os.path.join("models", "asl_classifier.pkl")
DIGIT_MODEL_PATH = os.path.join("models", "digit_classifier.pkl")
WORD_MODEL_PATH = os.path.join("models", "merged_dynamic_classifier.h5")

# Timers & Cooldowns
LETTER_COOLDOWN = 2.0          # Seconds between letter speech outputs
WORD_COOLDOWN = 2.5            # Seconds between dynamic word predictions/speech
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
    base_x, base_y, base_z = landmark_list[0]  # wrist as origin
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmark_list]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    return [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]

def extract_holistic_features(holistic_results):
    """Extracts 225-dim normalized features from MediaPipe Holistic results."""
    if holistic_results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.left_hand_landmarks.landmark])
        wrist = lh[0].copy()
        lh -= wrist
        max_val = np.max(np.abs(lh))
        if max_val > 0:
            lh /= max_val
        lh_flat = lh.flatten()
    else:
        lh_flat = np.zeros(63)

    if holistic_results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.right_hand_landmarks.landmark])
        wrist = rh[0].copy()
        rh -= wrist
        max_val = np.max(np.abs(rh))
        if max_val > 0:
            rh /= max_val
        rh_flat = rh.flatten()
    else:
        rh_flat = np.zeros(63)

    if holistic_results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.pose_landmarks.landmark])
        shoulder_mid = (pose[11] + pose[12]) / 2.0
        pose -= shoulder_mid
        max_val = np.max(np.abs(pose))
        if max_val > 0:
            pose /= max_val
        pose_flat = pose.flatten()
    else:
        pose_flat = np.zeros(99)

    return np.concatenate([lh_flat, rh_flat, pose_flat])

if __name__ == "__main__":
    letter_model = joblib.load(LETTER_MODEL_PATH) if os.path.exists(LETTER_MODEL_PATH) else None
    digit_model = joblib.load(DIGIT_MODEL_PATH) if os.path.exists(DIGIT_MODEL_PATH) else None

    # Load DynamicWordRecognizer safely
    word_recognizer = None
    if os.path.exists(WORD_MODEL_PATH):
        try:
            from dynamic_word_recognizer import DynamicWordRecognizer
            word_recognizer = DynamicWordRecognizer()
            print("Loaded DynamicWordRecognizer successfully.")
        except Exception as e:
            print(f"Could not initialize DynamicWordRecognizer: {e}")

    if letter_model is None and digit_model is None and word_recognizer is None:
        raise FileNotFoundError("No trained model files found in models/. Run training scripts first.")

    current_mode = "LETTER" if letter_model is not None else ("NUMBER" if digit_model is not None else "DYNAMIC_WORD")
    feature_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    # Paused / Idle state tracking
    # Paused mode halts model inference & TTS output to prevent false-positive recognitions
    # during natural resting periods, clothing adjustments, or scratching.
    is_paused = False

    # Prediction buffers
    prediction_buffer = deque(maxlen=10)
    holistic_buffer = deque(maxlen=30)  # Rolling 30-frame window for dynamic word mode

    # Multi-digit sequencing state machine
    number_buffer = ""
    candidate_digit = None
    digit_confirmed = False
    last_hand_seen_time = time.time()

    # Dynamic word state & session history log
    recognized_word_history = []
    last_word_prediction_time = 0
    latest_word_prediction = "None"
    latest_word_confidence = 0.0

    # MediaPipe Solutions
    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    holistic_detector = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = None
    for cam_idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                print(f"Webcam initialized on index {cam_idx}. Hotkeys: 'l'=Letter | 'n'=Number | 'd'=Dynamic Word | 'p'=Pause/Resume | 'c'=Clear | 'q'=Quit")
                break
            temp_cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError("No active webcam device found. Please check your camera connection.")

    last_thread_check = time.time()

    while cap.isOpened():
        if time.time() - last_thread_check > 5.0:
            last_thread_check = time.time()

        ret, detection_frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        raw_label = "No hand"
        confirmed_label = "No hand"

        if is_paused:
            # When Paused is ON:
            # 1. Skip model inference entirely (do NOT run through Random Forest or PyTorch LSTM).
            # 2. Render MediaPipe skeleton for visual user feedback only.
            # 3. Do NOT queue any TTS audio speech output.
            if current_mode in ["LETTER", "NUMBER"]:
                results = hands_detector.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(detection_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            elif current_mode == "DYNAMIC_WORD":
                holistic_results = holistic_detector.process(rgb_frame)
                if holistic_results.pose_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if holistic_results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if holistic_results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        else:
            # Active Recognition Modes
            if current_mode in ["LETTER", "NUMBER"]:
                active_model = letter_model if current_mode == "LETTER" else digit_model
                results = hands_detector.process(rgb_frame)

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

                        if len(prediction_buffer) == 10:
                            most_common, count = Counter(prediction_buffer).most_common(1)[0]
                            if count >= 7:
                                confirmed_label = most_common

                        current_time = time.time()

                        if current_mode == "LETTER":
                            if confirmed_label != "No hand":
                                if confirmed_label != last_spoken_label:
                                    speak_text(confirmed_label)
                                    last_spoken_label = confirmed_label
                                    last_spoken_time = current_time
                                elif (current_time - last_spoken_time > LETTER_COOLDOWN):
                                    speak_text(confirmed_label)
                                    last_spoken_time = current_time

                        elif current_mode == "NUMBER":
                            if confirmed_label != "No hand":
                                if confirmed_label != candidate_digit:
                                    candidate_digit = confirmed_label
                                    digit_confirmed = False
                                else:
                                    if not digit_confirmed:
                                        number_buffer += candidate_digit
                                        digit_confirmed = True
                else:
                    prediction_buffer.clear()
                    candidate_digit = None
                    digit_confirmed = False

                    if current_mode == "NUMBER" and len(number_buffer) > 0:
                        absent_duration = time.time() - last_hand_seen_time
                        if absent_duration >= HAND_ABSENT_TIMEOUT:
                            print(f"Number Sequence Complete: {number_buffer}")
                            speak_text(number_buffer)
                            number_buffer = ""

            elif current_mode == "DYNAMIC_WORD":
                holistic_results = holistic_detector.process(rgb_frame)

                # Draw upper body pose & hands keypoints
                if holistic_results.pose_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if holistic_results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if holistic_results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(detection_frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Extract 225-dim holistic feature vector and buffer
                frame_feat = extract_holistic_features(holistic_results)
                holistic_buffer.append(frame_feat)

                # Run prediction when 30 frames buffered & cooldown elapsed
                current_time = time.time()
                if len(holistic_buffer) == 30 and (current_time - last_word_prediction_time > WORD_COOLDOWN):
                    if word_recognizer is not None:
                        seq_array = np.array(holistic_buffer)
                        predicted_word, conf = word_recognizer.predict(seq_array)
                        if conf >= 0.5:
                            latest_word_prediction = predicted_word
                            latest_word_confidence = conf
                            last_word_prediction_time = current_time

                            # Append to session history log
                            entry = {"timestamp": current_time, "word": predicted_word, "confidence": conf}
                            recognized_word_history.append(entry)
                            print(f"[Dynamic Word Logged] '{predicted_word}' ({conf*100:.1f}%) | Total history: {len(recognized_word_history)} words")

                            # Audio TTS
                            speak_text(predicted_word)

        # Mirror frame for display
        display_frame = cv2.flip(detection_frame, 1)

        # UI Banners
        if is_paused:
            # Distinct Red/Yellow Paused Banner
            cv2.rectangle(display_frame, (10, 10), (620, 90), (0, 0, 180), -1)
            cv2.putText(display_frame, f"PAUSED -- press 'p' to resume [Mode: {current_mode}]", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, "Recognition & Audio Speech HALTED", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            if current_mode == "LETTER":
                mode_color = (0, 255, 0)
            elif current_mode == "NUMBER":
                mode_color = (255, 165, 0)
            else:
                mode_color = (255, 0, 255)  # Purple for Dynamic Word

            cv2.rectangle(display_frame, (10, 10), (620, 90), (0, 0, 0), -1)
            cv2.putText(display_frame, f"MODE: {current_mode} [l: Letter | n: Number | d: Dynamic | p: Pause]", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

            if current_mode == "LETTER":
                disp = confirmed_label if confirmed_label != "No hand" else (f"{raw_label}..." if raw_label != "No hand" else "No hand")
                cv2.putText(display_frame, f"Sign: {disp}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif current_mode == "NUMBER":
                disp = confirmed_label if confirmed_label != "No hand" else (f"{raw_label}..." if raw_label != "No hand" else "No hand")
                cv2.putText(display_frame, f"Sign: {disp} | Number: {number_buffer}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif current_mode == "DYNAMIC_WORD":
                buf_len = len(holistic_buffer)
                history_str = " ".join([h["word"] for h in recognized_word_history[-5:]])
                cv2.putText(display_frame, f"Word: {latest_word_prediction} ({latest_word_confidence*100:.0f}%) [Buff: {buf_len}/30]", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display_frame, f"History: {history_str}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("ASL Real-Time Translator", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_paused = not is_paused
            # Clear all prediction and sequence buffers when toggling paused state
            prediction_buffer.clear()
            holistic_buffer.clear()
            number_buffer = ""
            candidate_digit = None
            digit_confirmed = False

            if is_paused:
                print(f"[PAUSED] Recognition halted. Current mode preserved: {current_mode}. Press 'p' to resume.")
            else:
                print(f"[RESUMED] Recognition active. Resumed mode: {current_mode}.")

        elif key == ord('l'):
            current_mode = "LETTER"
            number_buffer = ""
            prediction_buffer.clear()
            holistic_buffer.clear()
            print("Switched to LETTER Mode.")
        elif key == ord('n'):
            if digit_model is None:
                print("Digit model missing!")
            else:
                current_mode = "NUMBER"
                number_buffer = ""
                prediction_buffer.clear()
                holistic_buffer.clear()
                print("Switched to NUMBER Mode.")
        elif key == ord('d'):
            if word_recognizer is None:
                print("Dynamic Word Recognizer not trained yet! Run training pipeline first.")
            else:
                current_mode = "DYNAMIC_WORD"
                prediction_buffer.clear()
                holistic_buffer.clear()
                print("Switched to DYNAMIC WORD Mode.")
        elif key == ord('c'):
            number_buffer = ""
            prediction_buffer.clear()
            holistic_buffer.clear()
            recognized_word_history.clear()
            print("Cleared buffers and word history.")

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    holistic_detector.close()

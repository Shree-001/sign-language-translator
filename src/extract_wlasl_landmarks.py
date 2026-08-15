import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

WLASL_JSON_PATH = os.path.join("data", "wlasl", "WLASL_v0.3.json")
CONFIG_PATH = os.path.join("data", "selected_words.json")
VIDEOS_DIR = os.path.join("data", "wlasl", "videos")
OUTPUT_DIR = os.path.join("data", "wlasl_sequences")

SEQUENCE_LENGTH = 30

def extract_frame_landmarks(holistic_results):
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

def process_video(video_path, mp_holistic):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_holistic.process(rgb)
        feat = extract_frame_landmarks(results)
        frame_features.append(feat)

    cap.release()

    if len(frame_features) == 0:
        return None

    frame_features = np.array(frame_features)
    n_frames = len(frame_features)

    if n_frames == SEQUENCE_LENGTH:
        final_seq = frame_features
    elif n_frames > SEQUENCE_LENGTH:
        indices = np.linspace(0, n_frames - 1, SEQUENCE_LENGTH, dtype=int)
        final_seq = frame_features[indices]
    else:
        pad_len = SEQUENCE_LENGTH - n_frames
        last_frame = frame_features[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        final_seq = np.vstack([frame_features, padding])

    return final_seq

def run_extraction():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found. Run scope_wlasl.py first.")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab = config["selected_words"]
    word_to_label = {word: i for i, word in enumerate(sorted(vocab))}
    
    with open(WLASL_JSON_PATH, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    tasks = []
    for entry in wlasl_data:
        gloss = entry.get("gloss", "").strip().lower()
        if gloss in vocab:
            for inst in entry.get("instances", []):
                video_id = inst.get("video_id")
                split = inst.get("split", "train")
                tasks.append((video_id, gloss, split))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_splits = {"train": [], "val": [], "test": []}
    y_splits = {"train": [], "val": [], "test": []}

    print(f"Initializing MediaPipe Holistic (Model Complexity 0)...")
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print(f"Extracting landmark sequences for {len(tasks)} videos across {len(vocab)} words...")
    processed_count = 0
    missing_count = 0

    for video_id, gloss, split in tqdm(tasks):
        video_file = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_file):
            missing_count += 1
            continue

        seq = process_video(video_file, mp_holistic)
        if seq is not None:
            X_splits[split].append(seq)
            y_splits[split].append(word_to_label[gloss])
            processed_count += 1

    mp_holistic.close()

    print(f"\nExtraction complete!")
    print(f"Successfully processed: {processed_count} videos | Missing video files: {missing_count}")

    for split in ["train", "val", "test"]:
        X_arr = np.array(X_splits[split])
        y_arr = np.array(y_splits[split])
        np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X_arr)
        np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), y_arr)
        print(f"  [{split}] X shape: {X_arr.shape}, y shape: {y_arr.shape}")

    label_mapping = {i: word for word, i in word_to_label.items()}
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

if __name__ == "__main__":
    run_extraction()

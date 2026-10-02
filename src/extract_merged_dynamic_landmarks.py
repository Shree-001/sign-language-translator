"""
src/extract_merged_dynamic_landmarks.py

Extracts MediaPipe Holistic landmark sequence features (Hands + Pose: 225 dimensions per frame)
for 8 high-value conversational ASL words from WLASL video clips. Applies multi-repetition /
sliding-window segmentation to extract multiple 30-frame sub-clip samples from long videos.
"""

import os
import json
import numpy as np
import cv2
import mediapipe as mp

WLASL_DIR = os.path.join("data", "wlasl")
JSON_PATH = os.path.join(WLASL_DIR, "WLASL_v0.3.json")
VIDEO_DIR = os.path.join(WLASL_DIR, "videos")
OUTPUT_DIR = os.path.join("data", "merged_dynamic_sequences")

# Target 8 concepts & mapped WLASL glosses
TARGET_MAP = {
    "hello": "hello",
    "thank_you": "thank you",
    "yes": "yes",
    "no": "no",
    "please": "please",
    "how_are_you": "how",
    "my_name": "name",
    "nice_to_meet_you": "meet"
}

SEQUENCE_LENGTH = 30
STRIDE = 12

def normalize_hand(landmarks):
    if not landmarks:
        return [0.0] * 63
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    base_x, base_y, base_z = pts[0]
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in pts]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    normalized = []
    for x, y, z in translated:
        normalized.extend([x / max_dist, y / max_dist, z / max_dist])
    return normalized

def normalize_pose(landmarks):
    if not landmarks:
        return [0.0] * 99
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    left_shoulder, right_shoulder = pts[11], pts[12]
    mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    mid_z = (left_shoulder[2] + right_shoulder[2]) / 2.0
    shoulder_dist = ((left_shoulder[0] - right_shoulder[0])**2 + (left_shoulder[1] - right_shoulder[1])**2 + (left_shoulder[2] - right_shoulder[2])**2)**0.5
    if shoulder_dist == 0:
        shoulder_dist = 1e-6
    normalized = []
    for x, y, z in pts:
        normalized.extend([(x - mid_x) / shoulder_dist, (y - mid_y) / shoulder_dist, (z - mid_z) / shoulder_dist])
    return normalized

def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        lh = normalize_hand(results.left_hand_landmarks)
        rh = normalize_hand(results.right_hand_landmarks)
        pose = normalize_pose(results.pose_landmarks)
        
        frame_features = lh + rh + pose
        frames.append(frame_features)
    cap.release()
    return np.array(frames, dtype=np.float32)

def resample_sequence(seq, target_len=30):
    curr_len = len(seq)
    if curr_len == target_len:
        return seq
    indices = np.linspace(0, curr_len - 1, target_len).astype(int)
    return seq[indices]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(JSON_PATH, "r") as f:
        wlasl_data = json.load(f)
    
    wlasl_dict = {item["gloss"]: item["instances"] for item in wlasl_data}
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
    
    label_to_idx = {label: i for i, label in enumerate(TARGET_MAP.keys())}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    
    X_samples = []
    y_samples = []
    meta_info = []
    
    print("--- Extracting Landmark Sequences & Multi-Repetition Sub-clips ---")
    
    summary_counts = {label: 0 for label in TARGET_MAP.keys()}
    source_video_counts = {label: 0 for label in TARGET_MAP.keys()}

    for target_label, gloss in TARGET_MAP.items():
        instances = wlasl_dict.get(gloss, [])
        video_count = 0
        clip_count = 0
        
        for inst in instances:
            vid = inst["video_id"]
            video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            if not os.path.exists(video_path):
                continue
            
            print(f"Processing {target_label} ({vid}.mp4)...", flush=True)
            raw_seq = process_video(video_path, holistic)
            if len(raw_seq) == 0:
                continue
            
            video_count += 1
            
            # Sub-clip slicing for videos > 35 frames
            if len(raw_seq) <= 35:
                sub_seq = resample_sequence(raw_seq, SEQUENCE_LENGTH)
                X_samples.append(sub_seq)
                y_samples.append(label_to_idx[target_label])
                meta_info.append({"word": target_label, "video_id": vid, "source": "WLASL"})
                clip_count += 1
            else:
                # Sliding window with STRIDE
                start_idx = 0
                while start_idx + SEQUENCE_LENGTH <= len(raw_seq):
                    sub_seq = raw_seq[start_idx : start_idx + SEQUENCE_LENGTH]
                    X_samples.append(sub_seq)
                    y_samples.append(label_to_idx[target_label])
                    meta_info.append({"word": target_label, "video_id": vid, "source": "WLASL"})
                    clip_count += 1
                    start_idx += STRIDE
                    
        summary_counts[target_label] = clip_count
        source_video_counts[target_label] = video_count
        print(f"Word: {target_label:18s} (gloss: '{gloss}') -> {video_count:2d} videos processed -> {clip_count:3d} sub-clip samples extracted", flush=True)

    holistic.close()
    
    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples, dtype=np.int32)
    
    print(f"\nTotal Dataset Extracted: X shape = {X.shape}, y shape = {y.shape}", flush=True)
    
    np.save(os.path.join(OUTPUT_DIR, "X_merged.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_merged.npy"), y)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Dataset successfully saved to {OUTPUT_DIR}/", flush=True)

if __name__ == "__main__":
    main()

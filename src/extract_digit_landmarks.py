"""
src/extract_digit_landmarks.py

VIVA EXPLANATION — DIGIT FEATURE EXTRACTION & NORMALIZATION
----------------------------------------------------------
1. MediaPipe 3D Landmark Extraction: Extracts 21 key spatial joints (63 floats) per hand image,
   isolating geometric hand posture for digits 0-9.
2. Wrist-Relative Normalization & Max Scale Invariance: Translates wrist (joint 0) to origin (0,0,0)
   and scales by max Euclidean joint distance, making features invariant to hand position and distance.
"""

import os
import cv2
import pandas as pd
import mediapipe as mp

DATA_DIR = os.path.join("data", "asl_digits")
OUTPUT_CSV = os.path.join("data", "digit_landmarks.csv")
MAX_IMAGES_PER_CLASS = 300

def normalize_landmarks(landmark_list):
    # Translate wrist landmark to origin (0,0,0) and scale by max joint distance
    base_x, base_y, base_z = landmark_list[0]
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmark_list]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    return [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Digit dataset directory missing: {DATA_DIR}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    data_rows = []

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()], key=int)
    print(f"Extracting landmarks for digit classes {classes} (max {MAX_IMAGES_PER_CLASS} images/class)...")

    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES_PER_CLASS]
        extracted_count = 0

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                raw_landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
                norm_landmarks = normalize_landmarks(raw_landmarks)
                row = []
                for x, y, z in norm_landmarks:
                    row.extend([x, y, z])
                row.append(class_name)
                data_rows.append(row)
                extracted_count += 1

        print(f"Digit '{class_name}': Processed {len(image_files)} images -> Extracted {extracted_count} landmark sets.")

    hands.close()

    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDigit landmark extraction complete! Saved {len(df)} rows to {OUTPUT_CSV}")

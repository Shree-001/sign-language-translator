"""
src/extract_landmarks.py

VIVA EXPLANATION — WHY LANDMARKS INSTEAD OF RAW PIXELS?
------------------------------------------------------
1. Dimensionality Reduction: Raw images contain 120,000+ pixels filled with background noise,
   varying lighting, skin tones, and camera artifacts.
2. MediaPipe Hands extracts 21 key structural hand joints (63 x,y,z spatial coordinates), isolating
   pure geometric hand configuration.
3. Lightweight & Fast: Reduces dataset feature dimension from ~120k to 63 tabular floats.
   This enables lightweight CPU real-time inference without needing high-end GPUs.
"""

import os
import cv2
import pandas as pd
import mediapipe as mp

DATA_DIR = os.path.join("data", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_CSV = os.path.join("data", "landmarks.csv")
MAX_IMAGES_PER_CLASS = 300

# Initialize MediaPipe Hands solution in static image mode
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Define column names: x0, y0, z0, ..., x20, y20, z20, label
columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
data_rows = []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset path not found: {DATA_DIR}")

classes = sorted(os.listdir(DATA_DIR))
print(f"Extracting landmarks for {len(classes)} classes (max {MAX_IMAGES_PER_CLASS} images/class)...")

def normalize_landmarks(landmark_list):
    # landmark_list: list of 21 (x, y, z) tuples from MediaPipe
    base_x, base_y, base_z = landmark_list[0]  # wrist as the origin
    translated = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmark_list]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6
    normalized = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]
    return normalized

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES_PER_CLASS]
    extracted_count = 0

    for img_name in image_files:
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Convert BGR (OpenCV default format) to RGB (MediaPipe requirement)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # Extract normalized (x, y, z) coordinates if a hand was detected
        if results.multi_hand_landmarks:
            raw_landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
            norm_landmarks = normalize_landmarks(raw_landmarks)
            row = []
            for x, y, z in norm_landmarks:
                row.extend([x, y, z])
            row.append(class_name)
            data_rows.append(row)
            extracted_count += 1

    print(f"Class '{class_name}': Processed {len(image_files)} images -> Extracted {extracted_count} landmark sets.")

hands.close()

# Save extracted feature dataset to CSV
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nExtraction complete! Saved {len(df)} rows to {OUTPUT_CSV}")

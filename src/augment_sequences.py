import os
import json
import numpy as np

DATA_DIR = os.path.join("data", "wlasl100_sequences")
OUTPUT_X_AUG = os.path.join(DATA_DIR, "X_train_augmented.npy")
OUTPUT_Y_AUG = os.path.join(DATA_DIR, "y_train_augmented.npy")

def spatial_jitter(sequence, noise_level=0.02):
    """Adds small random noise to non-zero landmark coordinates."""
    noise = np.random.normal(0, noise_level, sequence.shape).astype(np.float32)
    # Don't add noise to zero-padded features
    mask = (sequence != 0).astype(np.float32)
    return sequence + (noise * mask)

def random_scaling(sequence, scale_range=(0.9, 1.1)):
    """Scales landmark positions by a random factor."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (sequence * scale).astype(np.float32)

def horizontal_mirroring(sequence):
    """Flips x-coordinates (every 3rd element x in (x,y,z))."""
    seq_flipped = sequence.copy()
    # Left hand x: 0..63 (0, 3, 6...)
    # Right hand x: 63..126 (63, 66, 69...)
    # Pose x: 126..225 (126, 129, 132...)
    x_indices = list(range(0, 225, 3))
    seq_flipped[:, x_indices] = -seq_flipped[:, x_indices]
    return seq_flipped.astype(np.float32)

def temporal_resample(sequence, speed_factor=0.9):
    """Resamples 30-frame sequence to simulate faster or slower gesture execution."""
    orig_len = sequence.shape[0]  # 30
    target_len = int(orig_len * speed_factor)
    if target_len <= 5:
        return sequence.copy()

    indices = np.linspace(0, orig_len - 1, target_len, dtype=int)
    resampled = sequence[indices]

    if target_len == orig_len:
        return resampled
    elif target_len > orig_len:
        return resampled[:orig_len]
    else:
        pad_len = orig_len - target_len
        last_frame = resampled[-1:]
        padding = np.repeat(last_frame, pad_len, axis=0)
        return np.vstack([resampled, padding]).astype(np.float32)

def augment_dataset():
    x_train_path = os.path.join(DATA_DIR, "X_train.npy")
    y_train_path = os.path.join(DATA_DIR, "y_train.npy")

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print(f"Error: {x_train_path} not found. Run extract_wlasl100_landmarks.py first.")
        return

    X_train = np.load(x_train_path).astype(np.float32)
    y_train = np.load(y_train_path).astype(np.int64)

    print(f"Original Training Set: X={X_train.shape}, y={y_train.shape}")

    X_aug_list = [X_train]
    y_aug_list = [y_train]

    # Augmentation Pass 1: Spatial Jitter + Scaling
    X_aug1 = np.array([random_scaling(spatial_jitter(seq, 0.02)) for seq in X_train])
    X_aug_list.append(X_aug1)
    y_aug_list.append(y_train)

    # Augmentation Pass 2: Horizontal Mirroring + Jitter
    X_aug2 = np.array([horizontal_mirroring(spatial_jitter(seq, 0.015)) for seq in X_train])
    X_aug_list.append(X_aug2)
    y_aug_list.append(y_train)

    # Augmentation Pass 3: Speed 0.9x + Jitter
    X_aug3 = np.array([temporal_resample(spatial_jitter(seq, 0.02), 0.9) for seq in X_train])
    X_aug_list.append(X_aug3)
    y_aug_list.append(y_train)

    # Augmentation Pass 4: Speed 1.1x + Scaling
    X_aug4 = np.array([temporal_resample(random_scaling(seq, (0.92, 1.08)), 1.1) for seq in X_train])
    X_aug_list.append(X_aug4)
    y_aug_list.append(y_train)

    X_train_augmented = np.concatenate(X_aug_list, axis=0)
    y_train_augmented = np.concatenate(y_aug_list, axis=0)

    print("="*60)
    print(f"AUGMENTATION COMPLETE (5x Total Expansion):")
    print(f"  Original Training Samples : {len(X_train)}")
    print(f"  Augmented Training Samples: {len(X_train_augmented)}")
    print(f"  Final X shape             : {X_train_augmented.shape}")
    print(f"  Final y shape             : {y_train_augmented.shape}")
    print("="*60)

    np.save(OUTPUT_X_AUG, X_train_augmented)
    np.save(OUTPUT_Y_AUG, y_train_augmented)
    print(f"Saved augmented training dataset to {OUTPUT_X_AUG}")

if __name__ == "__main__":
    augment_dataset()

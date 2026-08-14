# Real-Time ASL Sign Language Translator (Letters & Digits)

![Version](https://img.shields.io/badge/version-1.1.4-blue.svg)

A lightweight, real-time American Sign Language (ASL) translator built with **MediaPipe Hands**, **Scikit-Learn (RandomForestClassifier)**, **OpenCV**, and **pyttsx3**. Recognized signs are converted into visual text overlays and spoken aloud via offline text-to-speech.

See [CHANGELOG.md](file:///c:/Users/shrin/sign-language-translator/CHANGELOG.md) for version history and release notes.

---

## Current Features & Status (v1.1.4)

- **Dual Model Architecture**:
  - **Alphabet Classifier** (`models/asl_classifier.pkl`): Recognizes letters `A–Z`, `space`, `del` (**99.52% accuracy**).
  - **Digit Classifier** (`models/digit_classifier.pkl`): Recognizes numbers `0–9` (**99.43% accuracy**).
  - Segregated classifiers eliminate visual hand-shape collisions (e.g. `1` vs `D`, `2` vs `V`, `0` vs `O`).
- **10-Frame Majority-Vote Stability Filter**:
  - Maintains a rolling 10-frame prediction buffer (`collections.deque(maxlen=10)`).
  - Requires \(\ge 7/10\) frame agreement before confirming gestures, filtering out raw classification jitter and preventing speech over-triggering.
- **Multi-Digit Sequencing State Machine**:
  - **Hold-Steady Confirmation**: Appends digit to number buffer once confirmed by majority-vote.
  - **2.0s Hand-Absent Sequence Completion**: Automatically speaks full accumulated number (e.g., `"237"`) when hand is removed for 2 seconds and resets buffer.
- **Hand-Relative Scale & Translation Normalization**: Translates wrist joint to origin \((0,0,0)\) and scales by maximum 3D joint distance for distance and camera framing invariance.
- **Automatic Handedness Mirroring**: Detects Left/Right hand via `multi_handedness` and mirrors Right-hand x-coordinates (`-x`) into Left-hand training coordinate space.
- **Thread-Safe SAPI5 TTS Engine**: Background queue worker thread with `pythoncom.CoInitialize()` and per-utterance engine lifecycle (`pyttsx3.init()` / `del engine`) ensuring audio plays for every prediction on Windows.
- **Clean Selfie Mirror UX**: OpenCV text overlays (`cv2.putText`) render left-to-right (un-mirrored) on top of the mirrored display feed.

---

## Project Structure

```
sign-language-translator/
├── data/
│   ├── landmarks.csv               # Extracted normalized alphabet feature dataset
│   └── digit_landmarks.csv         # Extracted normalized digit feature dataset
├── models/
│   ├── asl_classifier.pkl          # Trained Random Forest model (Alphabet)
│   └── digit_classifier.pkl        # Trained Random Forest model (Digits 0-9)
├── src/
│   ├── extract_landmarks.py        # Alphabet landmark feature extractor
│   ├── train_classifier.py         # Alphabet Random Forest classifier trainer
│   ├── extract_digit_landmarks.py  # Digit landmark feature extractor
│   ├── train_digit_classifier.py   # Digit Random Forest classifier trainer
│   └── realtime_predict.py         # Real-time webcam inference application
├── .gitignore
├── CHANGELOG.md
├── PROJECT_DOCUMENTATION.md
├── README.md
├── requirements.txt
└── VERSION
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Shree-001/sign-language-translator.git
cd sign-language-translator

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Real-Time Translator

```bash
python src/realtime_predict.py
```

### 3. Live Application Controls

| Keypress | Action |
| :--- | :--- |
| **`l`** | Switch to **Letter Mode** (`A–Z`, `space`, `del`) |
| **`n`** | Switch to **Number Mode** (`0–9` multi-digit sequencing) |
| **`c`** | Clear current in-progress number buffer |
| **`q`** | Quit application |

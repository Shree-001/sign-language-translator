# Real-Time ASL Sign Language Translator (Sign-to-Text & Text-to-Sign)

![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)

A lightweight, bidirectional American Sign Language (ASL) translator built with **MediaPipe Hands**, **Scikit-Learn (RandomForestClassifier)**, **OpenCV**, and **pyttsx3**. Offers both real-time **Sign-to-Text** webcam translation and interactive **Text-to-Sign** reverse translation.

See [CHANGELOG.md](file:///c:/Users/shrin/sign-language-translator/CHANGELOG.md) for version history and release notes.

---

## Features & Modes (v1.2.0)

### 1. Sign-to-Text Mode (`src/realtime_predict.py`)
- **Dual Model Architecture**:
  - **Alphabet Classifier** (`models/asl_classifier.pkl`): Recognizes letters `A–Z`, `space`, `del` (**99.52% accuracy**).
  - **Digit Classifier** (`models/digit_classifier.pkl`): Recognizes numbers `0–9` (**99.43% accuracy**).
- **10-Frame Majority-Vote Stability Filter**: Requires \(\ge 7/10\) frame agreement before confirming gestures, preventing speech over-triggering.
- **Multi-Digit Sequencing State Machine**: Accumulates steady digits into a number buffer and auto-speaks full numbers after 2.0s of hand absence.
- **Hand-Relative Normalization & Handedness Mirroring**: Distance and camera framing invariant with automatic Right-to-Left hand coordinate mirroring.
- **Thread-Safe SAPI5 TTS Engine**: Background queue worker thread with `pythoncom.CoInitialize()` and per-utterance engine lifecycle (`pyttsx3.init()` / `del engine`).

### 2. Text-to-Sign Mode (`src/text_to_sign.py`)
- **Bidirectional Translation**: Converts typed text or sentences into a sequential ASL sign image slideshow (fingerspelling style).
- **Deterministic Image Lookup**: Maps characters `A–Z`, `0–9`, and `spaces` to sample dataset images.
- **Sequential Display & Speech**: Plays sign slideshow at ~1s/frame in a single OpenCV window with `Sign: X` overlays, followed by spoken audio synthesis.

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
│   ├── realtime_predict.py         # Live webcam Sign-to-Text inference app
│   └── text_to_sign.py             # Interactive Text-to-Sign reverse translator
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

### 2. Run Sign-to-Text Translator (Webcam)

```bash
python src/realtime_predict.py
```
- **`l`**: Letter Mode (`A–Z`, `space`, `del`)
- **`n`**: Number Mode (`0–9` multi-digit sequencing)
- **`c`**: Clear number buffer
- **`q`**: Quit

### 3. Run Text-to-Sign Translator (Interactive)

```bash
python src/text_to_sign.py
```
- Type any word or phrase (e.g. `ASL 2026`) to watch the sign image slideshow and hear audio synthesis. Type `exit` to quit.

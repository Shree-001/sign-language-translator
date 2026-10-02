# Real-Time ASL Sign Language Translator (Sign-to-Text, Text-to-Sign & Dynamic Word Mode)

![Version](https://img.shields.io/badge/version-1.3.0-blue.svg)

A lightweight, multi-modal American Sign Language (ASL) translator built with **MediaPipe (Hands & Holistic)**, **Scikit-Learn (RandomForestClassifier)**, **PyTorch (LSTM)**, **OpenCV**, and **pyttsx3**. Offers real-time **Letter Mode**, **Number Mode**, **Dynamic Word Mode**, **Text-to-Sign** reverse translation, and a **Paused/Idle Mode**.

Designed as the input foundation for a planned sentence-assembly AI assistant system.

See [CHANGELOG.md](file:///c:/Users/shrin/sign-language-translator/CHANGELOG.md) for version history and release notes.

---

## Features & Modes (v1.3.0)

### 1. Dynamic Word Mode (`src/dynamic_word_recognizer.py` & `'d'` Keypress)
- **High-Value Conversational Vocabulary**: Recognizes 8 core words/phrases (`hello`, `thank_you`, `yes`, `no`, `please`, `how_are_you`, `my_name`, `nice_to_meet_you`).
- **MediaPipe Holistic Feature Extraction**: Extracts 225 3D spatial coordinates per frame (Left Hand 63, Right Hand 63, Pose 99) with wrist and shoulder normalization.
- **Multi-Repetition Sub-Clip Dataset**: Extracted 235 30-frame landmark sequences using sliding-window segmentation (30-frame window, 12-frame stride) across source videos.
- **PyTorch LSTM Classifier**: 2-layer LSTM sequence model trained with spatial jitter, scaling, horizontal mirroring, and balanced class weights (**89.36% test accuracy**, peak test accuracy **95.74%**).
- **Decoupled Architecture**: `DynamicWordRecognizer` is completely decoupled from the OpenCV UI, allowing direct import by sentence-assembly & AI assistant modules.
- **Session History Logging**: Appends recognized words to `recognized_word_history` with timestamps.

### 2. Sign-to-Text Mode (`src/realtime_predict.py`)
- **Alphabet Classifier** (`models/asl_classifier.pkl`): Recognizes letters `A–Z`, `space`, `del` (**99.52% accuracy**).
- **Digit Classifier** (`models/digit_classifier.pkl`): Recognizes numbers `0–9` (**99.43% accuracy**).
- **10-Frame Majority-Vote Stability Filter**: Requires \(\ge 7/10\) frame agreement before confirming gestures, preventing speech over-triggering.
- **Multi-Digit Sequencing State Machine**: Accumulates steady digits into a number buffer and auto-speaks full numbers after 2.0s of hand absence.

### 3. Text-to-Sign Mode (`src/text_to_sign.py`)
- **Bidirectional Translation**: Converts typed text into a sequential ASL sign image slideshow with text overlays and spoken voice.

### 4. Paused / Idle Mode (`'p'` Keypress)
- Halts gesture recognition and audio speech to prevent false positives when resting hands or adjusting clothing.

---

## Project Structure

```
sign-language-translator/
├── data/
│   ├── landmarks.csv                   # Extracted normalized alphabet feature dataset
│   ├── digit_landmarks.csv             # Extracted normalized digit feature dataset
│   └── merged_dynamic_sequences/       # Extracted 225-dim 30-frame dynamic word sequences
├── models/
│   ├── asl_classifier.pkl              # Trained Random Forest model (Alphabet)
│   ├── digit_classifier.pkl            # Trained Random Forest model (Digits 0-9)
│   ├── merged_dynamic_classifier.pth   # Trained PyTorch LSTM model (8-Word Dynamic Mode)
│   └── merged_label_map.json           # Label mapping for dynamic word classifier
├── src/
│   ├── extract_landmarks.py            # Alphabet landmark feature extractor
│   ├── train_classifier.py             # Alphabet Random Forest classifier trainer
│   ├── extract_digit_landmarks.py      # Digit landmark feature extractor
│   ├── train_digit_classifier.py       # Digit Random Forest classifier trainer
│   ├── extract_merged_dynamic_landmarks.py # Dynamic 225-dim landmark sequence extractor
│   ├── train_merged_dynamic_classifier.py   # PyTorch LSTM dynamic word classifier trainer
│   ├── dynamic_word_recognizer.py      # Standalone, decoupled Dynamic Word Recognizer
│   ├── realtime_predict.py             # Live webcam multi-mode inference app
│   └── text_to_sign.py                 # Interactive Text-to-Sign reverse translator
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

### 2. Run Real-Time Translator (Webcam)

```bash
python src/realtime_predict.py
```
- **`d`**: Switch to **Dynamic Word Mode** (`hello`, `thank_you`, `yes`, `no`, `please`, `how_are_you`, `my_name`, `nice_to_meet_you`)
- **`l`**: Switch to **Letter Mode** (`A–Z`, `space`, `del`)
- **`n`**: Switch to **Number Mode** (`0–9` multi-digit sequencing)
- **`p`**: Toggle **Paused / Idle Mode**
- **`c`**: Clear number buffer & word history log
- **`q`**: Quit

### 3. Run Text-to-Sign Translator (Interactive)

```bash
python src/text_to_sign.py
```

# Real-Time ASL Sign Language Translator (Sign-to-Text & Text-to-Sign)

![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)

A lightweight, bidirectional American Sign Language (ASL) translator built with **MediaPipe Hands & Holistic**, **PyTorch (LSTM)**, **Scikit-Learn (RandomForest)**, **OpenCV**, and **pyttsx3**. Offers real-time **Sign-to-Text** webcam translation, **Official WLASL100 Dynamic Sign Recognition (49% Top-1 / 72% Top-5 Test Accuracy)**, and interactive **Text-to-Sign** reverse translation. Designed as input infrastructure for a voice & sign assistant layer.

See [CHANGELOG.md](file:///c:/Users/shrin/sign-language-translator/CHANGELOG.md) for version history and release notes.

---

## Features & Modes (v1.4.0)

### 1. Multi-Mode Sign-to-Text (`src/realtime_predict.py`)
- **Tri-Model Architecture**:
  - **Alphabet Classifier** (`models/asl_classifier.pkl`): Recognizes letters `A–Z`, `space`, `del` (**99.52% accuracy**). Press `'l'` to activate.
  - **Digit Classifier** (`models/digit_classifier.pkl`): Recognizes numbers `0–9` (**99.43% accuracy**). Press `'n'` to activate.
  - **Dynamic Word Recognizer** (`models/dynamic_word_classifier_v2.pth`): PyTorch 2-layer LSTM sequence model trained on the **Official WLASL100 Benchmark Subset** (100 word classes) with 5x sequence data augmentation and balanced class weighting (**49.00% Top-1 / 72.00% Top-5 Test Accuracy**). Press `'d'` to activate.
- **Standalone Recognizer Module (`src/dynamic_word_recognizer.py`)**: Decoupled `DynamicWordRecognizer` class with `predict(sequence) -> (word, confidence)` for easy reuse in sentence-assembly and intent-classification modules.
- **Session History Logging**: Recognized words are automatically logged to `recognized_word_history` with timestamps for downstream assistant ingestion.
- **10-Frame Majority-Vote Stability Filter**: Requires \(\ge 7/10\) frame agreement before confirming static gestures.
- **Multi-Digit Sequencing State Machine**: Accumulates steady digits into a number buffer and auto-speaks full numbers after 2.0s of hand absence.
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
│   ├── landmarks.csv                   # Extracted normalized alphabet feature dataset
│   ├── digit_landmarks.csv             # Extracted normalized digit feature dataset
│   ├── selected_words_wlasl100.json    # Scoped WLASL100 100-word benchmark configuration
│   └── wlasl100_sequences/             # 30-frame Holistic sequence arrays (raw & 5x augmented)
├── models/
│   ├── asl_classifier.pkl              # Trained Random Forest model (Alphabet)
│   ├── digit_classifier.pkl            # Trained Random Forest model (Digits 0-9)
│   ├── dynamic_word_classifier_v2.pth  # Trained PyTorch 2-layer LSTM model (WLASL100)
│   └── wlasl100_labels.json            # Label mapping for 100-word dynamic classifier
├── src/
│   ├── extract_landmarks.py            # Alphabet landmark feature extractor
│   ├── train_classifier.py             # Alphabet Random Forest classifier trainer
│   ├── extract_digit_landmarks.py      # Digit landmark feature extractor
│   ├── train_digit_classifier.py       # Digit Random Forest classifier trainer
│   ├── scope_wlasl100.py               # Official WLASL100 vocabulary scoping script
│   ├── extract_wlasl100_landmarks.py   # MediaPipe Holistic sequence extractor (WLASL100)
│   ├── augment_sequences.py            # 5x spatial/temporal/mirroring sequence data augmentor
│   ├── train_wlasl100_classifier.py    # PyTorch LSTM trainer (class balancing & Top-1/5)
│   ├── dynamic_word_recognizer.py      # Decoupled, reusable DynamicWordRecognizer class
│   ├── realtime_predict.py             # Live webcam Sign-to-Text inference app ('l'/'n'/'d')
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

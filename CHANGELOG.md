# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-08-16 - feat
- **Official WLASL100 Benchmark Rebuild**: Replaced initial 30-word hand-picked subset with the official benchmark **WLASL100 dataset** (100 glosses, 2,038 videos) to eliminate majority-class degeneration caused by data scarcity and imbalanced sample counts.
- **5x Sequence Data Augmentation** (`src/augment_sequences.py`): Expanded training sequence dataset from 748 to **3,740 sequence samples** using spatial noise jitter ($\pm 2\%$), coordinate scaling ($0.9\times\text{--}1.1\times$), horizontal mirroring ($x \rightarrow -x$), and temporal speed resampling ($0.9\times\text{--}1.1\times$).
- **Class-Weighted PyTorch LSTM Training** (`src/train_wlasl100_classifier.py`): Trained 2-layer LSTM sequence model (`hidden_dim=128`, `dropout=0.4`) using `balanced` class weighting in `CrossEntropyLoss`.
- **WLASL Literature Benchmark Alignment**: Evaluated model performance on unseen WLASL100 test set:
  - **Top-1 Test Accuracy**: **49.00%** (surpassing published Pose-GRU WLASL literature baseline of 46.25%).
  - **Top-5 Test Accuracy**: **72.00%**.
- Updated `DynamicWordRecognizer` (`src/dynamic_word_recognizer.py`) and `realtime_predict.py` to load `models/dynamic_word_classifier_v2.pth` and `models/wlasl100_labels.json`.

## [1.3.0] - 2026-08-15 - feat
- Added **Dynamic (Word-Level) Sign Recognition Mode** (`'d'`) using MediaPipe Holistic (225 keypoint features for hands and upper-body pose) and a PyTorch 2-layer LSTM sequence classifier.
- Integrated **WLASL Dataset** vocabulary (30 target words including weather queries like `tell`, `weather`, `time`, `what`, `today`, `hot`, `cold`, `rain`, and core conversational glue words like `hello`, `yes`, `no`, `please`, `help`, `thanks`, `sorry`).
- Created a standalone, decoupled `DynamicWordRecognizer` module (`src/dynamic_word_recognizer.py`) with `predict(sequence) -> (word, confidence)` to serve as the input foundation for a future sentence-assembly and AI assistant layer.
- Added session history logging (`recognized_word_history`) in `realtime_predict.py` with timestamps to log recognized word sequences across webcam sessions.

## [1.2.0] - 2026-08-14 - feat
- Added **Text-to-Sign Mode** (`src/text_to_sign.py`) for reverse ASL translation (text-to-fingerspelling slideshow with text overlay and speech synthesis).
- Parses letters (A–Z), digits (0–9), and spaces to look up corresponding sign images deterministically from training dataset folders.
- Displays characters in sequence at ~1 frame/second in a single OpenCV window with `Sign: X` banner overlays, followed by spoken audio synthesis.

## [1.1.4] - 2026-08-14 - fix
- Applied a rolling 10-frame majority-vote filter (`collections.deque(maxlen=10)`) requiring \(\ge 7/10\) frame agreement before confirming predictions.
- Eliminated per-frame raw classification jitter from over-triggering 8–10 speech calls during a single held sign.
- Updated on-screen text overlays to display confirmed predictions and stabilization indicators (`Sign: X (stabilizing...)`).
- Standardized majority-vote stability filtering across both Letter Mode and Number Mode state machines.

## [1.1.3] - 2026-08-14 - fix
- Fixed TTS silence after first utterance by creating a new `pyttsx3` engine per utterance instead of reusing one engine instance across the worker thread's lifetime (SAPI5 internal state doesn't reset on engine reuse).

## [1.1.2] - 2026-08-14 - fix
- Initialized COM apartment on background worker thread using `pythoncom.CoInitialize()` and `pythoncom.CoUninitialize()` to fix silent TTS failures on Windows SAPI5.

## [1.1.1] - 2026-08-14 - fix
- Fixed horizontally mirrored on-screen text in OpenCV display window by separating `detection_frame` and `display_frame`, ensuring `cv2.putText()` is executed after `cv2.flip()`.

## [1.1.0] - 2026-08-14 - feat
- Added Number Mode (0–9) with a separate `digit_classifier.pkl` (99.43% test accuracy) to prevent hand-shape collisions (e.g. 1≈D, 2≈V, 0≈O).
- Added 'n' / 'l' mode toggling shortcuts and 'c' manual clear shortcut.
- Added multi-digit sequencing state machine with 1.0s hold-steady digit confirmation and 2.0s hand-absence auto-completion.

## [1.0.1] - 2026-08-10 - fix
- Added hand-relative wrist-origin translation and max-distance scaling normalization to eliminate distance and camera framing sensitivity (boosted letter test accuracy to 99.52%).
- Added automatic handedness detection and Right-to-Left x-axis mirroring to support both hands.

## [1.0.0] - 2026-08-09 - feat
- Initial release: Real-time static ASL Alphabet recognition (A–Z, space, del) using MediaPipe 21 hand landmarks, Scikit-Learn RandomForestClassifier, OpenCV, and offline pyttsx3 Text-to-Speech.

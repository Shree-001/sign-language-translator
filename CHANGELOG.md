# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

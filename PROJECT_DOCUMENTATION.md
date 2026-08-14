# Real-Time ASL Sign Language Translator (Sign-to-Text & Text-to-Sign)
## Complete Technical Overview, Multi-Digit State Machine, & Defense Guide

---

## 1. Executive Summary

The **Real-Time ASL Sign Language Translator** is an intelligent, bidirectional computer vision and machine learning system that translates between American Sign Language (ASL) and English. It supports:
1. **Sign-to-Text Mode**: Live webcam translation of alphabet letters (`A–Z`, `space`, `del`) and numbers (`0–9`) into text overlays and spoken audio.
2. **Text-to-Sign Mode**: Interactive reverse translation converting typed sentences into sequential fingerspelled sign image slideshows and spoken voice.

To solve gesture overlap between identical hand shapes (e.g., `1` vs `D`, `2` vs `V`, `0` vs `O`), the project uses a **Dual Model Architecture** with mode toggling (`'l'` / `'n'`) and a **Multi-Digit Sequencing State Machine**.

---

## 2. System Architecture & Dual Model Workflow

```
[MODE 1: SIGN-TO-TEXT WEBCAM TRANSLATOR]
+-----------------------------------------------------------------------------------+
|                                 LIVE WEBCAM FEED                                  |
|                          (Raw Unflipped detection_frame)                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              MEDIAPIPE HANDS PIPELINE                             |
|       (21 3D Joint Keypoints -> Wrist Translation -> Max Scale Normalization)      |
+-----------------------------------------------------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
      [MODE: LETTER  (Press 'l')]             [MODE: NUMBER  (Press 'n')]
      +-------------------------+             +-------------------------+
      |  asl_classifier.pkl     |             |  digit_classifier.pkl   |
      |  Random Forest (A-Z)    |             |  Random Forest (0-9)    |
      +-------------------------+             +-------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     10-FRAME MAJORITY-VOTE STABILITY FILTER                       |
|   collections.deque(maxlen=10) requires >= 7/10 frame agreement before confirming |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    THREAD-SAFE SPEECH WORKER & COM INITIALIZATION                 |
|   pythoncom.CoInitialize() initializes STA COM apartment. speech_worker creates   |
|   a fresh pyttsx3 engine per utterance inside queue loop and releases it (del).   |
+-----------------------------------------------------------------------------------+

[MODE 2: TEXT-TO-SIGN REVERSE TRANSLATOR]
+-----------------------------------------------------------------------------------+
|               USER TERMINAL INPUT (e.g. "HELLO 123" -> Parse Chars)               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|           DATASET IMAGE LOOKUP (A-Z -> asl_alphabet, 0-9 -> asl_digits)          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|         SEQUENTIAL OPENCV SLIDESHOW (~1s/frame) WITH OVERLAY & TTS AUDIO          |
+-----------------------------------------------------------------------------------+
```

---

## 3. Multi-Digit Sequencing State Machine

```
               [ User signs digit ]
                        |
                        v
        +-------------------------------+
        | Candidate Digit Detected?     |
        +-------------------------------+
            |                       |
       (Same Digit)           (Digit Changed)
            |                       |
            v                       v
  +--------------------+   +--------------------+
  | 7/10 Majority Vote |   | Reset Hold Buffer  |
  +--------------------+   +--------------------+
     |              |
   (Yes)           (No)
     |              |
     v              +-----> [ Wait for stabilization ]
+-------------------------+
| Append to Number Buffer |
|  (e.g., "2" -> "23")    |
+-------------------------+
            |
            v
+----------------------------------+
| Hand Absent for >= 2.0 Seconds?  |
+----------------------------------+
     |
   (Yes)
     v
+----------------------------------+
| Speak Full Number via pyttsx3    |
| (e.g., "Two Hundred Thirty-Seven")|
| & Reset Buffer                   |
+----------------------------------+
```

---

## 4. Retrained Classifier Performance Metrics

### Letter Classifier (`models/asl_classifier.pkl`)
- **Evaluation Accuracy**: **99.52%** across 1,470 test samples.

### Digit Classifier (`models/digit_classifier.pkl`)
- **Evaluation Accuracy**: **99.43%** across 527 test samples (from 2,632 extracted landmark rows).

---

## 5. Key Bug Fixes & Technical Refinements

| Bug | Technical Root Cause | Resolution |
| :--- | :--- | :--- |
| **Bidirectional Communication** | Previous pipeline only supported Sign-to-Text webcam translation. | **Text-to-Sign Mode (`src/text_to_sign.py`)**: Added interactive text input parsing that displays sequential fingerspelled sign images with overlays and speaks final phrases. |
| **Over-Triggering Speech Jitter** | Classifier runs at ~30 FPS; tiny raw per-frame classification fluctuations pushed 8–10 speech calls during a single held sign. | **10-Frame Majority-Vote Filter**: Maintained a rolling `deque(maxlen=10)` prediction buffer. Predictions are confirmed only when \(\ge 7/10\) frames agree, ensuring one clean speech call per intended sign. |
| **Mirrored On-Screen Text** | `cv2.putText()` was called on raw frame before `cv2.flip()`, rendering text mirrored horizontally. | **Flip-Then-Draw Order**: Skeleton drawn on raw `detection_frame`, then `display_frame = cv2.flip(detection_frame, 1)` is generated, and `cv2.putText()` calls render on `display_frame` so text reads left-to-right. |
| **First-Only Speech Output** | SAPI5 driver event loop state in `pyttsx3` does not reset cleanly after `runAndWait()` when reusing a single engine object across multiple utterances. | **Per-Utterance Engine Lifecycle**: `pythoncom.CoInitialize()` is invoked once per thread, while `engine = pyttsx3.init()` and `del engine` execute per utterance inside the queue loop. |

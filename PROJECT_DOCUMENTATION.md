# Real-Time ASL Sign Language Translator (Letters & Digits)
## Complete Technical Overview, Multi-Digit State Machine, & Defense Guide

---

## 1. Executive Summary

The **Real-Time ASL Sign Language Translator** is an intelligent computer vision and machine learning system that translates both American Sign Language (ASL) **alphabet letters (A–Z, space, del)** and **digit numbers (0–9)** into text and spoken audio in real time using a standard webcam.

To solve the challenge of visual gesture overlap between identical hand shapes (e.g., `1` vs `D`, `2` vs `V`, `0` vs `O`), the project uses a **Dual Model Architecture** with real-time mode toggling (`'l'` for Letter Mode, `'n'` for Number Mode) and a **Multi-Digit Sequencing State Machine**.

---

## 2. System Architecture & Dual Model Workflow

```
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
                  |                                       |
                  v                                       v
      +-------------------------+             +-------------------------+
      | 2.0s Speech Cooldown    |             | Multi-Digit State       |
      | Single-Letter Output    |             | Machine (Hold & Buffer) |
      +-------------------------+             +-------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       FLIP-THEN-DRAW UI RENDER PIPELINE                           |
|  1. Draw Hand Skeleton onto raw frame -> 2. display_frame = cv2.flip(frame, 1)    |
|  3. Draw text overlays (cv2.putText) on display_frame so text reads left-to-right  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                  THREAD-SAFE TTS QUEUE & PER-UTTERANCE ENGINE                     |
|   pythoncom.CoInitialize() initializes STA COM apartment. speech_worker creates   |
|   a fresh pyttsx3 engine per utterance inside queue loop and releases it (del).   |
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
  | Hold Timer >= 1.0s?|   | Reset Hold Timer   |
  +--------------------+   +--------------------+
     |              |
   (Yes)           (No)
     |              |
     v              +-----> [ Wait for hold ]
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

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        18
           1       1.00      1.00      1.00        60
           2       1.00      0.96      0.98        53
           3       0.97      1.00      0.98        60
           4       1.00      1.00      1.00        59
           5       1.00      1.00      1.00        60
           6       1.00      0.98      0.99        46
           7       1.00      1.00      1.00        58
           8       0.98      1.00      0.99        54
           9       1.00      1.00      1.00        59

    accuracy                           0.99       527
```

---

## 5. Key Bug Fixes & Technical Refinements

| Bug | Technical Root Cause | Resolution |
| :--- | :--- | :--- |
| **Mirrored On-Screen Text** | `cv2.putText()` was called on raw frame before `cv2.flip()`, rendering text mirrored horizontally. | **Flip-Then-Draw Order**: Skeleton drawn on raw `detection_frame`, then `display_frame = cv2.flip(detection_frame, 1)` is generated, and `cv2.putText()` calls render on `display_frame` so text reads left-to-right. |
| **First-Only Speech Output** | SAPI5 driver event loop state in `pyttsx3` does not reset cleanly after `runAndWait()` when reusing a single engine object across multiple utterances. | **Per-Utterance Engine Lifecycle**: `pythoncom.CoInitialize()` is invoked once per thread, while `engine = pyttsx3.init()` and `del engine` execute per utterance inside the queue loop. |

---

## 6. Viva Defense & Interview Cheat Sheet

#### Q1: Why do you instantiate a fresh `pyttsx3` engine per utterance instead of reusing one instance?
> **Answer**: On Windows, SAPI5 drivers do not reset internal event loop state reliably after `engine.runAndWait()` is called on a reused engine object across multiple predictions. By calling `engine = pyttsx3.init()` per queued utterance and releasing it with `del engine` inside a COM-initialized background thread (`pythoncom.CoInitialize()`), every prediction plays audibly without stuttering or stopping after the first phrase.

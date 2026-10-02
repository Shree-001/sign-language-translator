# Real-Time ASL Sign Language Translator (Sign-to-Text, Text-to-Sign & Dynamic Word Assistant Foundation)
## Complete Technical Overview, Multi-Digit State Machine, & Defense Guide

---

## 1. Executive Summary

The **Real-Time ASL Sign Language Translator** is an intelligent, multi-modal computer vision and machine learning system that translates between American Sign Language (ASL) and English across four operational modes:
1. **Dynamic Word Mode**: Real-time continuous gesture recognition for 8 high-value conversational words/phrases (`hello`, `thank_you`, `yes`, `no`, `please`, `how_are_you`, `my_name`, `nice_to_meet_you`) using MediaPipe Holistic & a PyTorch LSTM sequence model (**89.36% test accuracy**, peak test accuracy **95.74%**). Designed as the input foundation for a planned AI sentence-assembly assistant.
2. **Sign-to-Text Letter Mode**: Live webcam translation of alphabet letters (`A–Z`, `space`, `del`) (**99.52% accuracy**).
3. **Sign-to-Text Number Mode**: Live multi-digit number translation (`0–9`) (**99.43% accuracy**).
4. **Text-to-Sign Mode**: Interactive reverse translation converting typed sentences into sequential fingerspelled sign image slideshows and spoken voice.

---

## 2. System Architecture & Multi-Modal Workflow

```
[MODE 1: DYNAMIC WORD RECOGNITION & AI ASSISTANT FOUNDATION]
+-----------------------------------------------------------------------------------+
|                                 LIVE WEBCAM FEED                                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              MEDIAPIPE HOLISTIC PIPELINE                          |
|    (Left Hand 63 + Right Hand 63 + Upper Body Pose 99 = 225 Features per Frame)   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         ROLLING 30-FRAME FEATURE BUFFER                           |
|       (deque(maxlen=30) passed to DynamicWordRecognizer every ~1 second)           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       STANDALONE DYNAMIC WORD RECOGNIZER                          |
|       (models/merged_dynamic_classifier.pth -> 2-Layer PyTorch LSTM Model)         |
+-----------------------------------------------------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
      +-------------------------------+       +-------------------------------+
      | Session Word History Logging  |       | Non-Blocking SAPI5 Audio      |
      | (Timestamped Session Log)     |       | Speech Output (pyttsx3)       |
      +-------------------------------+       +-------------------------------+
```

---

## 3. Dataset Multi-Repetition Sub-Clip Segmentation

| Word Concept | Gloss | Source Videos | Sub-Clip Samples Extracted |
| :--- | :--- | :---: | :---: |
| `hello` | `hello` | 4 | 14 |
| `thank_you` | `thank you` | 7 | 26 |
| `yes` | `yes` | 12 | 37 |
| `no` | `no` | 11 | 38 |
| `please` | `please` | 7 | 31 |
| `how_are_you` | `how` | 9 | 37 |
| `my_name` | `name` | 6 | 22 |
| `nice_to_meet_you` | `meet` | 9 | 30 |
| **Total** | | **65** | **235** |

---

## 4. Key Technical Refinements & Decoupled Architecture

1. **Decoupled `DynamicWordRecognizer` Class**:
   - `src/dynamic_word_recognizer.py` contains `DynamicWordRecognizer` with `predict(landmark_sequence) -> (word, confidence)`.
   - Completely independent of OpenCV UI or webcam loops, enabling direct import by sentence-assembly, intent classification, and API assistant modules.
2. **Session History Persistence**:
   - Recognized dynamic words are appended to `recognized_word_history` with timestamps and confidence scores.

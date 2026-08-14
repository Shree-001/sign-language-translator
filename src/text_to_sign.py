"""
src/text_to_sign.py

VIVA EXPLANATION — REVERSE TEXT-TO-SIGN TRANSLATION PIPELINE
------------------------------------------------------------
1. Reverse ASL Translation: Translates typed text/sentences into a sequential ASL sign image slideshow,
   enabling bidirectional communication (Sign-to-Text & Text-to-Sign).
2. Deterministic Image Lookup: Maps characters A-Z to data/asl_alphabet_train/ and digits 0-9 to
   data/asl_digits/, selecting the first valid sample image per class.
3. Thread-Safe Speech Synthesis: Speaks the full word aloud via pyttsx3 inside a COM-initialized thread
   after the visual sign slideshow completes.
"""

import os
import cv2
import time
import threading
import pythoncom

ALPHABET_DIR = os.path.join("data", "asl_alphabet_train", "asl_alphabet_train")
DIGIT_DIR = os.path.join("data", "asl_digits")

def speak_text(text):
    def _speak():
        pythoncom.CoInitialize()
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            del engine
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            pythoncom.CoUninitialize()
    threading.Thread(target=_speak, daemon=True).start()

def get_sign_image_path(char):
    char_upper = char.upper()
    
    # 1. Letters A-Z
    if 'A' <= char_upper <= 'Z':
        char_dir = os.path.join(ALPHABET_DIR, char_upper)
        if os.path.exists(char_dir):
            files = sorted([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                return os.path.join(char_dir, files[0])

    # 2. Digits 0-9
    elif char.isdigit():
        char_dir = os.path.join(DIGIT_DIR, char)
        if os.path.exists(char_dir):
            files = sorted([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                return os.path.join(char_dir, files[0])

    # 3. Space character (using space class from alphabet dataset for simplicity)
    elif char == ' ':
        space_dir = os.path.join(ALPHABET_DIR, "space")
        if os.path.exists(space_dir):
            files = sorted([f for f in os.listdir(space_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                return os.path.join(space_dir, files[0])

    return None

def main():
    print("=" * 60)
    print("      ASL Text-to-Sign Translator (Fingerspelling Mode)")
    print("=" * 60)

    while True:
        user_input = input("\nEnter text to translate to ASL signs (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting Text-to-Sign Translator.")
            break
        if not user_input:
            continue

        print(f"\nTranslating '{user_input}' into ASL signs...")

        interrupted = False
        for char in user_input:
            img_path = get_sign_image_path(char)

            if img_path is None:
                if char != ' ':
                    print(f"No sign available for: '{char}'")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image for: '{char}'")
                continue

            # Resize image to a standard 400x400 display frame
            display_img = cv2.resize(img, (400, 400))

            # Overlay sign banner text (matching realtime_predict style)
            cv2.rectangle(display_img, (10, 10), (380, 60), (0, 0, 0), -1)
            display_char = "SPACE" if char == ' ' else char.upper()
            cv2.putText(display_img, f"Sign: {display_char}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

            cv2.imshow("ASL Text-to-Sign Translator", display_img)

            # Hold frame for 1 second (~1000ms)
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                print("Translation interrupted by user.")
                interrupted = True
                break

        cv2.destroyAllWindows()

        if not interrupted:
            print(f"Finished visual translation. Speaking: '{user_input}'")
            speak_text(user_input)

if __name__ == "__main__":
    main()

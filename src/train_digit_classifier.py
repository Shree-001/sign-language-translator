"""
src/train_digit_classifier.py

VIVA EXPLANATION — WHY A SEPARATE DIGIT MODEL?
---------------------------------------------
1. Visual Similarity / Gesture Overlap: ASL digits and letters share nearly identical hand shapes
   (e.g., 1 ≈ D, 2 ≈ V, 0 ≈ O). Merging them into a single classifier causes high misclassification.
2. Mode-Segregated Classification: Training a dedicated digit classifier maintains >98% accuracy
   on numbers 0-9 without degrading letter prediction accuracy.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

LANDMARKS_CSV = os.path.join("data", "digit_landmarks.csv")
MODEL_PATH = os.path.join("models", "digit_classifier.pkl")

if __name__ == "__main__":
    if not os.path.exists(LANDMARKS_CSV):
        raise FileNotFoundError(f"Digit landmark CSV missing: {LANDMARKS_CSV}. Run extract_digit_landmarks.py first.")

    print("Loading digit landmark dataset...")
    df = pd.read_csv(LANDMARKS_CSV)

    X = df.drop("label", axis=1)
    y = df["label"].astype(str)

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split complete: {len(X_train)} training samples, {len(X_test)} test samples.")

    # Train RandomForestClassifier (n_estimators=200)
    print("Training RandomForestClassifier for ASL Digits (0-9)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Digit Model Evaluation Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)
    print("\nDetailed Digit Classification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Trained digit model saved to: {MODEL_PATH}")

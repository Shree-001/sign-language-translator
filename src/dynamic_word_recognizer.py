import os
import json
import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL_PATH = os.path.join("models", "dynamic_word_classifier_v2.pth")
DEFAULT_LABELS_PATH = os.path.join("models", "wlasl100_labels.json")

class DynamicWordLSTM100(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

class DynamicWordRecognizer:
    """
    Standalone, reusable recognizer module for dynamic (word-level) ASL signs.
    Decoupled from webcam and UI logic for easy integration into future
    sentence-assembly, intent-classification, and AI assistant layers.
    Loaded with WLASL100 100-word benchmark model.
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels mapping not found at: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
        
        self.label_mapping = {int(k): v for k, v in raw_labels.items()}
        num_classes = len(self.label_mapping)

        self.model = DynamicWordLSTM100(input_dim=225, hidden_dim=128, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, landmark_sequence):
        """
        Predicts word label and confidence score from a 30-frame landmark sequence.
        
        Parameters:
            landmark_sequence (np.ndarray): Shape (30, 225) or (1, 30, 225)
            
        Returns:
            tuple: (word_string, confidence_float)
        """
        seq = np.array(landmark_sequence, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)

        if seq.shape[1:] != (30, 225):
            raise ValueError(f"Expected input shape (1, 30, 225), but got {seq.shape}")

        tensor_in = torch.from_numpy(seq)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        word = self.label_mapping.get(best_idx, "unknown")

        return word, confidence

if __name__ == "__main__":
    print("Testing DynamicWordRecognizer initialization with WLASL100...")
    try:
        recognizer = DynamicWordRecognizer()
        dummy_input = np.zeros((30, 225))
        word, conf = recognizer.predict(dummy_input)
        print(f"Sanity test prediction: Word='{word}', Confidence={conf:.4f}")
    except Exception as e:
        print(f"Note: {e}")

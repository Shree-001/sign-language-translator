"""
src/dynamic_word_recognizer.py

Standalone, decoupled Dynamic Word Recognizer module for 8-class ASL word recognition.
Uses PyTorch LSTM model trained on 225-dim MediaPipe Holistic features.
Can be imported by realtime_predict.py or downstream sentence-assembly & AI assistant modules.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

class PyTorchLSTMClassifier(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=64, num_classes=8, num_layers=2):
        super(PyTorchLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class DynamicWordRecognizer:
    def __init__(self, model_path="models/merged_dynamic_classifier.pth", label_map_path="models/merged_label_map.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}. Run train_merged_dynamic_classifier.py first.")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map file missing: {label_map_path}.")
            
        with open(label_map_path, "r") as f:
            label_data = json.load(f)
            
        self.idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
        self.label_to_idx = label_data["label_to_idx"]
        num_classes = len(self.idx_to_label)
        
        self.model = PyTorchLSTMClassifier(input_dim=225, hidden_dim=64, num_classes=num_classes, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, landmark_sequence) -> tuple:
        """
        Input: landmark_sequence of shape (30, 225)
        Returns: (predicted_word: str, confidence: float)
        """
        seq_arr = np.array(landmark_sequence, dtype=np.float32)
        if seq_arr.shape != (30, 225):
            return ("Unknown", 0.0)
            
        input_tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0) # (1, 30, 225)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            confidence = float(probs[top_idx].item())
            
        return (self.idx_to_label[top_idx], confidence)

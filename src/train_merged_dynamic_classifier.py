"""
src/train_merged_dynamic_classifier.py

Trains a 2-layer PyTorch LSTM sequence classifier on the extracted 8-class 225-dim landmark dataset.
Applies data augmentation (spatial jitter, scaling, mirroring) and balanced class weighting.
Saves trained model weights to models/merged_dynamic_classifier.pth and label map to models/merged_label_map.json.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = os.path.join("data", "merged_dynamic_sequences")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "merged_dynamic_classifier.pth")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "merged_label_map.json")

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

def augment_sequence(seq):
    augmented = seq.copy()
    
    # 1. Spatial Jitter
    jitter = np.random.normal(0, 0.01, size=augmented.shape).astype(np.float32)
    augmented += jitter
    
    # 2. Random Scale Factor
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale
    
    # 3. Horizontal Mirroring (flip x coordinates)
    if np.random.rand() > 0.5:
        augmented[:, 0::3] = -augmented[:, 0::3]
        
    return augmented

def augment_dataset(X, y, multiplier=4):
    X_aug = []
    y_aug = []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        for _ in range(multiplier):
            X_aug.append(augment_sequence(X[i]))
            y_aug.append(y[i])
    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int64)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    X_path = os.path.join(DATA_DIR, "X_merged.npy")
    y_path = os.path.join(DATA_DIR, "y_merged.npy")
    label_path = os.path.join(DATA_DIR, "label_map.json")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Extracted landmark sequences not found. Run extract_merged_dynamic_landmarks.py first.")
        
    X = np.load(X_path)
    y = np.load(y_path)
    with open(label_path, "r") as f:
        label_data = json.load(f)
        
    idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
    num_classes = len(idx_to_label)
    
    print(f"Loaded Dataset: X = {X.shape}, y = {y.shape}, Classes = {num_classes}")
    
    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Base Train split: {X_train.shape[0]} samples | Base Test split: {X_test.shape[0]} samples")
    
    # Augment Training set
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, multiplier=4)
    print(f"Augmented Training set: {X_train_aug.shape[0]} samples")
    
    # Compute Class Weights
    classes = np.unique(y_train_aug)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_aug)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class Weights:", class_weights)
    
    train_dataset = TensorDataset(torch.tensor(X_train_aug, dtype=torch.float32), torch.tensor(y_train_aug, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = PyTorchLSTMClassifier(input_dim=225, hidden_dim=64, num_classes=num_classes, num_layers=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0.0
    epochs = 80
    
    print("\n--- Training PyTorch LSTM Classifier ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on Test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        test_acc = correct / total
        
        if test_acc > best_accuracy or epoch == epochs:
            best_accuracy = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}% (Best: {best_accuracy*100:.2f}%)", flush=True)

    # Load Best Model for Final Evaluation
    best_model = PyTorchLSTMClassifier(input_dim=225, hidden_dim=64, num_classes=num_classes, num_layers=2)
    best_model.load_state_dict(torch.load(MODEL_PATH))
    best_model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = best_model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())
            
    final_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    
    print(f"\n==========================================", flush=True)
    print(f"  Merged 8-Word Classifier Test Accuracy: {final_acc * 100:.2f}%", flush=True)
    print(f"==========================================", flush=True)
    
    target_names = [idx_to_label[i] for i in range(num_classes)]
    
    print("\nClassification Report:", flush=True)
    print(classification_report(all_targets, all_preds, target_names=target_names), flush=True)
    
    print("\nConfusion Matrix:", flush=True)
    print(confusion_matrix(all_targets, all_preds), flush=True)
    
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_data, f, indent=2)
        
    print(f"\nModel & Label Map saved to {MODEL_PATH} and {LABEL_MAP_PATH}", flush=True)

if __name__ == "__main__":
    main()

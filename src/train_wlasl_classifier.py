import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = os.path.join("data", "wlasl_sequences")
MODEL_SAVE_PATH = os.path.join("models", "dynamic_word_classifier.pth")
LABELS_SAVE_PATH = os.path.join("models", "wlasl_labels.json")

class DynamicWordLSTM(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=30):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        last_out = out[:, -1, :]  # Take last frame output
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def train_model():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)

    with open(os.path.join(DATA_DIR, "label_mapping.json"), "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    os.makedirs("models", exist_ok=True)
    with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    num_classes = len(label_mapping)
    seq_len = X_train.shape[1]
    feature_dim = X_train.shape[2]

    print(f"Dataset Loaded:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}")
    print(f"  Classes: {num_classes}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = DynamicWordLSTM(input_dim=feature_dim, hidden_dim=128, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    epochs = 100

    print("\nStarting PyTorch LSTM Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total

        if (epoch % 5 == 0) or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Evaluate Best Model on Test Set
    best_model = DynamicWordLSTM(input_dim=feature_dim, hidden_dim=128, num_classes=num_classes)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    best_model.eval()

    X_test_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        logits = best_model(X_test_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    target_names = [label_mapping[str(i)] for i in range(num_classes)]

    print("\n" + "="*50)
    print(f"EVALUATION ON TEST SET (Accuracy: {acc*100:.2f}%):")
    print("="*50)
    labels_idx = list(range(num_classes))
    print(classification_report(y_test, preds, labels=labels_idx, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    train_model()

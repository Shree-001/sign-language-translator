import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = os.path.join("data", "wlasl100_sequences")
MODEL_SAVE_PATH = os.path.join("models", "dynamic_word_classifier_v2.pth")
LABELS_SAVE_PATH = os.path.join("models", "wlasl100_labels.json")

class DynamicWordLSTM100(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, 30, 128)
        last_step = out[:, -1, :]  # (batch, 128)
        x = self.fc1(last_step)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def evaluate_top_k(logits, targets, k=5):
    """Calculates top-k accuracy given logits tensor and target tensor."""
    top_k_preds = torch.topk(logits, k=k, dim=1).indices
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1).float().sum().item()
    return correct / targets.size(0)

def train_model():
    aug_x_path = os.path.join(DATA_DIR, "X_train_augmented.npy")
    aug_y_path = os.path.join(DATA_DIR, "y_train_augmented.npy")

    if os.path.exists(aug_x_path) and os.path.exists(aug_y_path):
        print(f"Loading augmented training dataset: {aug_x_path}")
        X_train = np.load(aug_x_path).astype(np.float32)
        y_train = np.load(aug_y_path).astype(np.int64)
    else:
        print("Augmented training set not found. Loading raw training dataset...")
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

    # Compute Balanced Class Weights
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    
    # Full weight array for all classes
    class_weights_array = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(unique_classes, weights):
        class_weights_array[c] = float(w)
    
    class_weights_tensor = torch.from_numpy(class_weights_array).float()

    print(f"\nWLASL100 Dataset Loaded:")
    print(f"  Train (Augmented): X={X_train.shape}, y={y_train.shape}")
    print(f"  Val              : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test             : X={X_test.shape}, y={y_test.shape}")
    print(f"  Classes          : {num_classes}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DynamicWordLSTM100(input_dim=feature_dim, hidden_dim=128, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_top1 = 0.0
    patience = 25
    patience_counter = 0
    epochs = 120

    print("\nStarting WLASL100 PyTorch LSTM Training (with Balanced Loss)...")
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
        val_loss = 0.0
        val_logits_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_logits_list.append(outputs)
                val_targets_list.append(batch_y)

        val_logits = torch.cat(val_logits_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        val_top1 = evaluate_top_k(val_logits, val_targets, k=1)
        val_top5 = evaluate_top_k(val_logits, val_targets, k=5)

        if (epoch % 5 == 0) or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Acc: {train_acc*100:.2f}% | Val Top-1: {val_top1*100:.2f}% | Val Top-5: {val_top5*100:.2f}%")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Evaluate Best Model on Test Set
    best_model = DynamicWordLSTM100(input_dim=feature_dim, hidden_dim=128, num_classes=num_classes)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    best_model.eval()

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    with torch.no_grad():
        test_logits = best_model(X_test_tensor)
        probs = torch.softmax(test_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    test_top1 = evaluate_top_k(test_logits, y_test_tensor, k=1)
    test_top5 = evaluate_top_k(test_logits, y_test_tensor, k=5)

    target_names = [label_mapping[str(i)] for i in range(num_classes)]

    print("\n" + "="*60)
    print(f"OFFICIAL WLASL100 TEST BENCHMARK RESULTS:")
    print(f"  TOP-1 ACCURACY: {test_top1*100:.2f}%")
    print(f"  TOP-5 ACCURACY: {test_top5*100:.2f}%")
    print("="*60)

    # Per-class accuracy calculation for Best & Worst 10 words
    class_accs = {}
    for c in range(num_classes):
        c_mask = (y_test == c)
        if np.sum(c_mask) > 0:
            c_acc = np.sum(preds[c_mask] == c) / np.sum(c_mask)
            class_accs[target_names[c]] = (c_acc, int(np.sum(c_mask)))

    sorted_accs = sorted(class_accs.items(), key=lambda item: item[1][0], reverse=True)

    print("\nTOP 10 BEST PERFORMING WORDS ON TEST SET:")
    print("-" * 50)
    for word, (acc_val, count) in sorted_accs[:10]:
        print(f" - {word:<15}: {acc_val*100:>5.1f}% accuracy ({count} test samples)")

    print("\nTOP 10 WORST PERFORMING WORDS ON TEST SET:")
    print("-" * 50)
    for word, (acc_val, count) in sorted_accs[-10:]:
        print(f" - {word:<15}: {acc_val*100:>5.1f}% accuracy ({count} test samples)")

if __name__ == "__main__":
    train_model()

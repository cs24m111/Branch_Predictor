import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm

# -------------------- DEVICE CONFIG --------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"🖥️ Using device: {device}")

# -------------------- MODEL --------------------
class CNNBranchPredictor(nn.Module):
    def __init__(self, table_size=2048, num_filters=32, history_len=200):
        super(CNNBranchPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(table_size, 1))
        self.fc = nn.Linear(history_len * num_filters, 1)

    def forward(self, x):
        x = x.unsqueeze(1)           # (batch, 1, table_size, history_len)
        x = self.conv1(x)            # (batch, num_filters, 1, history_len)
        x = torch.tanh(x)
        x = x.view(x.size(0), -1)    # Flatten
        x = self.fc(x)
        return torch.sigmoid(x)

# -------------------- HISTORY ENCODER --------------------
def encode_history(history, table_size=2048, history_len=200):
    matrix = np.zeros((table_size, history_len), dtype=np.float32)
    for i, (ip, opcode, direction) in enumerate(history[-history_len:]):
        ip7 = ip & 0x7F
        opcode3 = opcode & 0x07
        index = (((ip7 << 3) + opcode3) << 1) + direction
        index = index & (table_size - 1)
        matrix[index, i] = 1.0
    return matrix

# -------------------- STREAMING DATASET --------------------
class BranchHistoryIterableDataset(IterableDataset):
    def __init__(self, filepath, history_len=200, table_size=2048):
        self.filepath = filepath
        self.history_len = history_len
        self.table_size = table_size

    def __iter__(self):
        buffer = []
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                _, ip_hex, opcode, direction = parts
                try:
                    ip = int(ip_hex, 16)
                    opcode = int(opcode)
                    direction = int(direction)
                except ValueError:
                    continue

                buffer.append((ip, opcode, direction))

                # Skip until we have enough history
                if len(buffer) < self.history_len + 1:
                    continue

                history = buffer[-(self.history_len + 1):-1]
                label = buffer[-1][2]
                matrix = encode_history(history, self.table_size, self.history_len)
                yield torch.tensor(matrix), torch.tensor([label], dtype=torch.float32)

# -------------------- TRAINING FUNCTION --------------------
def train_model(model, loader, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        print(f"\n🌱 Epoch {epoch+1}/{epochs}")
        
        for X_batch, y_batch in tqdm(loader, desc="Training", unit="batch"):
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"📉 Average Loss: {avg_loss:.4f}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    dataset = BranchHistoryIterableDataset("../ChampSim/branch_history.log")
    loader = DataLoader(dataset, batch_size=16)

    model = CNNBranchPredictor()

    train_model(model, loader, epochs=10)

    # Save model
    torch.save(model.state_dict(), "cnn_branch_predictor.pth")
    print("✅ Model saved as cnn_branch_predictor.pth")

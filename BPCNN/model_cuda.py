import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm

# -------------------- DEVICE CONFIG --------------------
device = torch.device("cpu")
print(f"🖥️ Using device: {device}")

# -------------------- MODEL --------------------
class CNNBranchPredictor(nn.Module):
    def __init__(self, table_size=2048, num_filters=32, history_len=200):
        super(CNNBranchPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(table_size, 1))
        self.fc = nn.Linear(history_len * num_filters, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = x.view(x.size(0), -1)
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
    def __init__(self, filepath, history_len=200, table_size=2048, resume_line=0):
        self.filepath = filepath
        self.history_len = history_len
        self.table_size = table_size
        self.resume_line = resume_line
        self.last_line = resume_line

    def __iter__(self):
        buffer = []
        line_index = 0
        with open(self.filepath, 'r') as f:
            for line in f:
                line_index += 1
                if line_index < self.resume_line:
                    continue

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

                if len(buffer) < self.history_len + 1:
                    continue

                history = buffer[-(self.history_len + 1):-1]
                label = buffer[-1][2]
                self.last_line = line_index
                matrix = encode_history(history, self.table_size, self.history_len)
                yield torch.tensor(matrix), torch.tensor([label], dtype=torch.float32)

# -------------------- TRAINING FUNCTION --------------------
def train_model(model, dataset, loader, epochs=2, lr=1e-3, resume_epoch=0, resume_step=0, save_interval=10000):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    step = resume_step

    for epoch in range(resume_epoch, resume_epoch + epochs):
        total_loss = 0
        batch_count = 0
        print(f"\n🌱 Epoch {epoch+1}")

        pbar = tqdm(loader, unit="batch")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            total_loss += loss.item()
            batch_count += 1

            pbar.set_description(f"Step {step} | Epoch {epoch+1}")
            pbar.set_postfix(loss=loss.item())

            # 🔁 Save periodically
            if step % save_interval == 0:
                model_path = f"models/cnn_branch_predictor_step_{step}.pth"
                torch.save(model.state_dict(), model_path)
                state = {
                    "last_epoch": epoch,
                    "last_step": step,
                    "last_line": dataset.last_line
                }
                with open("states/resume_state.json", "w") as f:
                    json.dump(state, f)
                print(f"\n💾 Intermediate Save: {model_path} (line {dataset.last_line}, step {step})")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"📉 Avg Loss Epoch {epoch+1}: {avg_loss:.4f}")

        # 🔒 Final save for the epoch
        model_path = f"models/cnn_branch_predictor_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        with open("states/resume_state.json", "w") as f:
            json.dump({
                "last_epoch": epoch + 1,
                "last_step": step,
                "last_line": dataset.last_line
            }, f)
        print(f"✅ Model saved: {model_path} (line {dataset.last_line}, step {step})")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    log_path = "../ChampSim/branch_history.log"
    resume_epoch = 0
    resume_line = 0
    resume_step = 0

    if os.path.exists("states/resume_state.json"):
        with open("states/resume_state.json", "r") as f:
            state = json.load(f)
            resume_epoch = state.get("last_epoch", 0)
            resume_line = state.get("last_line", 0)
            resume_step = state.get("last_step", 0)
            print(f"🔁 Resuming from epoch {resume_epoch}, step {resume_step}, line {resume_line}")

    model = CNNBranchPredictor()
    model_path = f"models/cnn_branch_predictor_step_{resume_step}.pth" if resume_step else f"models/cnn_branch_predictor_epoch{resume_epoch}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"📦 Loaded model from {model_path}")

    dataset = BranchHistoryIterableDataset(log_path, resume_line=resume_line)
    loader = DataLoader(dataset, batch_size=16)

    train_model(model, dataset, loader, epochs=2, resume_epoch=resume_epoch, resume_step=resume_step, save_interval=10000)

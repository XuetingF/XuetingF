import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


# -------------------------
# Model Definition
# -------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Load tabular CSV
# (Domino will use YOUR CSV path)
# -------------------------

def load_tabular_csv(csv_path, feature_cols, target_col):
    df = pd.read_csv(csv_path)

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X, y)
    return dataset


# -------------------------
# Training Loop
# -------------------------

def train(model, dataloader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")


# -------------------------
# Main (Domino will modify)
# -------------------------

def main():
    # ⛔ THIS IS THE ONLY PART YOU CHANGE IN DOMINO ⛔
    csv_path = "/mnt/data/your_dataset.csv"     # <-- CHANGE ON DOMINO

    feature_cols = ["feature1", "feature2", "feature3"]   # <-- CHANGE
    target_col = "target"                                 # <-- CHANGE

    dataset = load_tabular_csv(csv_path, feature_cols, target_col)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLP(input_dim=len(feature_cols))
    train(model, dataloader)


if __name__ == "__main__":
    main()

"""
神经网络近似器 MLP回归模型，预测最大流数值
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from config import DEVICE, TORCH_DTYPE


class MaxFlowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h, dtype=TORCH_DTYPE))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1, dtype=TORCH_DTYPE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1)  # shape [batch]


def build_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle=True):
    x_tensor = torch.from_numpy(X).to(device=DEVICE, dtype=TORCH_DTYPE)
    y_tensor = torch.from_numpy(y).to(device=DEVICE, dtype=TORCH_DTYPE)
    ds = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_mlp(
    model: MaxFlowMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    lr: float,
    epochs: int
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_log = []
    model.train()

    for epoch in range(epochs):
        total_train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # test eval
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        model.train()

        train_log.append([epoch+1, avg_train_loss, avg_test_loss])
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | train_loss:{avg_train_loss:.4f} | test_loss:{avg_test_loss:.4f}")
    return train_log


def predict_mlp(model: MaxFlowMLP, X: np.ndarray):
    model.eval()
    x_tensor = torch.from_numpy(X).to(device=DEVICE, dtype=TORCH_DTYPE)
    with torch.no_grad():
        pred = model(x_tensor)
    return pred.detach().cpu().numpy()

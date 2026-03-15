import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# STEP 1 — Generate BALANCED synthetic data for training
print("Generating balanced data...")
X_balanced, y_balanced = make_classification(
    n_samples=100000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    weights=[0.5, 0.5],      # BALANCED
    flip_y=0.01,
    random_state=42
)

# Generate realistic imbalanced test set
X_imbal, y_imbal = make_classification(
    n_samples=200000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    weights=[0.9975, 0.0025],  # Real TalkingData distribution
    flip_y=0.001,
    random_state=99
)

feature_names = ['ip', 'app', 'device', 'os', 'channel',
                 'click_hour', 'click_day', 'click_dayofweek',
                 'ip_count', 'app_channel_count']

# STEP 2 — Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_balanced)
y_train = y_balanced

X_test = scaler.transform(X_imbal)
y_test = y_imbal

joblib.dump(scaler, 'scaler.pkl')
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# STEP 3 — Define Model & Optimizer
class FraudDetectionNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

class VarianceRMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9,
                 epsilon=1e-8, v_min=1e-10, v_max=10.0):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon,
                       v_min=v_min, v_max=v_max)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data)
                v = state['v']
                beta = group['beta']
                v_new = torch.clamp(
                    beta * v + (1 - beta) * grad ** 2,
                    group['v_min'], group['v_max']
                )
                state['v'] = v_new
                sigma_sq = torch.clamp(
                    v_new - (beta * v) ** 2,
                    min=group['epsilon']
                )
                p.data -= (group['lr'] /
                    torch.sqrt(sigma_sq + group['epsilon'])) * grad

# STEP 4 — Train
X_tr = torch.FloatTensor(X_train)
y_tr = torch.FloatTensor(y_train).unsqueeze(1)
dataset = TensorDataset(X_tr, y_tr)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = FraudDetectionNet(input_dim=10)
optimizer = VarianceRMSProp(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

best_loss = float('inf')
print("Starting training...")
for epoch in range(30):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/30 | Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'variance_rmsprop_model.pth')

print("Best model saved.")

# STEP 5 — Verify
model.load_state_dict(torch.load('variance_rmsprop_model.pth'))
model.eval()
with torch.no_grad():
    zero_input = torch.zeros(1, 10)
    p0 = model.predict_proba(zero_input).item()
    
    high_input = torch.ones(1, 10) * 2.0
    p_high = model.predict_proba(high_input).item()
    
    low_input = torch.ones(1, 10) * -2.0
    p_low = model.predict_proba(low_input).item()

print(f"Zero input → {p0:.4f}")
print(f"High input → {p_high:.4f}")
print(f"Low input  → {p_low:.4f}")

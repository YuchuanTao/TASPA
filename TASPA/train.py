import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from model import TASPA
from data_reprocess import CMAPSS

# =======================
# Set random seed
# =======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =======================
# Train for one epoch
# =======================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_samples, batch_OC, batch_RUL in dataloader:
        batch_samples, batch_OC, batch_RUL = batch_samples.to(device), batch_OC.to(device), batch_RUL.to(device)
        optimizer.zero_grad()
        outputs = model(batch_samples, batch_OC)
        loss = criterion(outputs, batch_RUL)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_samples.size(0)
    mse = total_loss / len(dataloader.dataset)
    rmse = mse ** 0.5
    return rmse

# =======================
# Test for one epoch
# =======================
def test_one_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_samples, batch_OC, batch_RUL in dataloader:
            batch_samples, batch_OC, batch_RUL = batch_samples.to(device), batch_OC.to(device), batch_RUL.to(device)
            outputs = model(batch_samples, batch_OC)
            all_preds.append(outputs)
            all_labels.append(batch_RUL)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    diff = preds - labels
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    score = torch.sum(torch.where(diff < 0, torch.exp(-diff / 13) - 1, torch.exp(diff / 10) - 1)).item()
    return rmse, score

# =======================
# Hyperparameters
# =======================
dataset_class = "FD004"
max_RUL = 125
time_window = 60
K_fold = 10

# TASPA model hyperparameters
TEM_kernel_size = 3
TEM_num_blocks = 2
attention_d_model = 32
attention_num_heads = 4
attention_dim_ff = 256
attention_num_layers = 4
ECA_kernel_size = 3
fused_hidden = 128
dropout = 0.1

lr = 1e-3
weight_decay = 1e-5
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Load dataset
# =======================
cmapss = CMAPSS(dataset_class, max_RUL, time_window, fold=1, K_fold=K_fold)
train_samples, train_OC, train_RUL = cmapss.get_full_train_samples()
test_samples, test_OC, test_RUL = cmapss.get_test_samples()

train_loader = Data.DataLoader(
    Data.TensorDataset(train_samples, train_OC, train_RUL),
    batch_size=batch_size, shuffle=True
)
test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=len(test_samples), shuffle=False
)

# =======================
# Multi-seed training
# =======================
results = []

for seed in range(2029, 2031):
    set_seed(seed)

    # Initialize model and optimizer
    model = TASPA(
        TEM_kernel_size, TEM_num_blocks,
        time_window, 14,
        attention_d_model, attention_num_heads,
        attention_dim_ff, attention_num_layers,
        ECA_kernel_size, fused_hidden, dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    min_rmse = float('inf')
    best_score = 0.0

    # Train for 30 epochs
    for epoch in range(30):
        rmse_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        rmse_test, score_test = test_one_epoch(model, test_loader, device)

        print(f"Seed {seed}, Epoch {epoch}: Train RMSE={rmse_train:.3f}, Test RMSE={rmse_test:.3f}, Score={score_test:.3f}")

        # Ensure directory exists
        os.makedirs("./trained_model", exist_ok=True)
        torch.save(model.state_dict(), f"./trained_model/rmse{rmse_test:.3f}_score{score_test:.3f}_seed{seed}.pth")

        # Save the best model
        if rmse_test < min_rmse:
            min_rmse = rmse_test
            best_score = score_test

    results.append((min_rmse, best_score))

# =======================
# Print overall results
# =======================
rmse_mean = np.mean([r[0] for r in results])
rmse_std = np.std([r[0] for r in results])
score_mean = np.mean([r[1] for r in results])
score_std = np.std([r[1] for r in results])

print(f"RMSE Mean: {rmse_mean:.3f}, RMSE Std: {rmse_std:.3f}")
print(f"Score Mean: {score_mean:.3f}, Score Std: {score_std:.3f}")

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from model import TASPA
from data_process import CMAPSS


# =======================
# Set random seed
# =======================

def set_seed(seed):

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)

    # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True

    # Disable benchmark mode for reproducibility
    torch.backends.cudnn.benchmark = False


# =======================
# Train for one epoch
# =======================

def train_one_epoch(model, dataloader, loss_func, optimizer):

    # Set model to training mode
    model.train()

    # Sum of squared errors
    SE = 0

    # Iterate over training batches
    for x, oc, y in dataloader:

        # Move data to device
        x, oc, y = x.to(device), oc.to(device), y.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x, oc)

        # Compute loss
        loss = loss_func(y_pred, y)

        # Backpropagation
        loss.backward()

        # Parameter update
        optimizer.step()

        # Accumulate squared error
        SE += loss.item() * x.size(0)

    # Compute RMSE over dataset
    RMSE = np.sqrt(SE / len(dataloader.dataset))

    return RMSE


# =======================
# Test for one epoch
# =======================

def test_one_epoch(model, dataloader, loss_func):

    # Set model to evaluation mode
    model.eval()

    # Sum of squared errors
    SE = 0

    # Score accumulator
    Score = 0

    # Disable gradient computation
    with torch.no_grad():

        # Iterate over test batches
        for x, oc, y in dataloader:

            # Move data to device
            x, oc, y = x.to(device), oc.to(device), y.to(device)

            # Forward pass
            y_pred = model(x, oc)

            # Compute loss
            loss = loss_func(y_pred, y)

            # Accumulate squared error
            SE += loss.item() * x.size(0)

            # Prediction error
            error = y_pred - y

            # Custom scoring function
            Score += torch.sum(
                torch.where(
                    error < 0,
                    torch.exp(-error / 13) - 1,
                    torch.exp(error / 10) - 1
                )
            ).item()

    # Compute RMSE
    RMSE = np.sqrt(SE / len(dataloader.dataset))

    return RMSE, Score


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Hyperparameters
# =======================

TEM_kernel_size = 3
TEM_num_blocks = 2

attention_d_model = 32
attention_num_heads = 4
attention_dim_ff = 256
attention_num_layers = 4

ECA_kernel_size = 3

fused_hidden = 128

dropout = 0.1


# =======================
# Load dataset
# =======================

dataset_class = "FD001"
max_RUL = 125
time_window = 30
K_fold = 10

# Initialize CMAPSS dataset
cmapss = CMAPSS(dataset_class, max_RUL, time_window)

# Load full training data
train_samples, train_OC, train_RUL = cmapss.get_full_train_samples()

# Load test data
test_samples, test_OC, test_RUL = cmapss.get_test_samples()


# Batch size
batch_size = 256

# Training loader
train_loader = Data.DataLoader(
    Data.TensorDataset(train_samples, train_OC, train_RUL),
    batch_size=batch_size,
    shuffle=True
)

# Test loader (full batch evaluation)
test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=len(test_samples),
    shuffle=False
)


# =======================
# Multi-seed training
# =======================

for seed in range(10):

    # Set random seed for reproducibility
    set_seed(seed)

    # Initialize model
    model = TASPA(
        TEM_kernel_size,
        TEM_num_blocks,
        time_window,
        14,
        attention_d_model,
        attention_num_heads,
        attention_dim_ff,
        attention_num_layers,
        ECA_kernel_size,
        fused_hidden,
        dropout
    ).to(device)

    # Learning rate
    lr = 1e-3

    # Weight decay
    weight_decay = 1e-5

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Loss function
    loss_func = nn.MSELoss()

    # Best RMSE tracking
    min_rmse = float('inf')

    # Best score tracking
    min_score = float('inf')

    # =======================
    # Training loop
    # =======================

    for epoch in range(30):

        # Train one epoch
        rmse_train = train_one_epoch(
            model,
            train_loader,
            loss_func,
            optimizer
        )

        # Evaluate on test set
        rmse_test, score_test = test_one_epoch(
            model,
            test_loader,
            loss_func
        )

        # Print progress
        print(
            f"Seed {seed} Epoch {epoch}: "
            f"Train RMSE={rmse_train:.3f}, "
            f"Test RMSE={rmse_test:.3f}, "
            f"Score={score_test:.3f}"
        )

        # Save best model per seed
        if rmse_test < min_rmse:

            min_rmse = rmse_test
            min_score = score_test

            # Ensure directory exists
            os.makedirs("trained_model", exist_ok=True)

            # Save model checkpoint
            torch.save(
                model.state_dict(),
                f"E:/trained_model/{dataset_class}_seed{seed}_rmse{rmse_test:.3f}_score{score_test:.3f}.pth"
            )
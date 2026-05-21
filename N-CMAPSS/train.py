import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from model import TASPA
from data_process import N_CMAPSS


# =======================
# Set random seed
# =======================

def set_seed(seed):

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch CPU random seed
    torch.manual_seed(seed)

    # PyTorch GPU random seed
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True

    # Disable benchmark for reproducibility
    torch.backends.cudnn.benchmark = False


# =======================
# Train for one epoch
# =======================

def train_one_epoch(model, dataloader, loss_func, optimizer):

    # Set model to training mode
    model.train()

    # Sum of squared errors
    SE = 0

    # Iterate through all training batches
    for x, oc, y in dataloader:

        # Move data to device
        x, oc, y = x.to(device), oc.to(device), y.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        y_pred = model(x, oc)

        # Compute loss
        loss = loss_func(y_pred, y)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate squared error
        SE += loss.item() * x.size(0)

    # Compute RMSE for training set
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

    # Score function accumulator
    Score = 0

    # Disable gradient computation
    with torch.no_grad():

        # Iterate through all test batches
        for x, oc, y in dataloader:

            # Move data to device
            x, oc, y = x.to(device), oc.to(device), y.to(device)

            # Forward propagation
            y_pred = model(x, oc)

            # Compute mean squared error
            loss = loss_func(y_pred, y)

            # Accumulate squared error
            SE += loss.item() * x.size(0)

            # Prediction error
            error = y_pred - y

            # Compute custom scoring function
            Score += torch.sum(
                torch.where(
                    error < 0,
                    torch.exp(-error / 13) - 1,
                    torch.exp(error / 10) - 1
                )
            ).item()

    # Compute RMSE
    RMSE = np.sqrt(SE / len(dataloader.dataset))

    # Compute average score
    Scorea = Score / len(dataloader.dataset)

    return RMSE, Scorea


# Select GPU if available
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

# Sliding window length
time_window = 70

# Dataset file path
file_path = 'E:/N-CMAPSS/N-CMAPSS_DS02-006.h5'

# Downsampling interval
sample_step = 10

# Initialize dataset processor
n_cmapss = N_CMAPSS(file_path, sample_step, time_window)

# Load training data
train_samples, train_OC, train_RUL = n_cmapss.get_train_data()

# Load test data
test_samples, test_OC, test_RUL = n_cmapss.get_test_data()

# Batch size
batch_size = 256

# Build training dataloader
train_loader = Data.DataLoader(
    Data.TensorDataset(train_samples, train_OC, train_RUL),
    batch_size=batch_size,
    shuffle=True
)

# Build test dataloader
test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=batch_size,
    shuffle=False
)


# Set random seed
set_seed(2026)

# Initialize TASPA model
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

# Weight decay coefficient
weight_decay = 1e-5

# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# Mean squared error loss
loss_func = nn.MSELoss()

# Best RMSE initialization
min_rmse = float('inf')

# Best score initialization
min_score = float('inf')

# Flag indicating whether NaN occurs during training
skip_seed = False


# =======================
# Training process
# =======================

# Train for 30 epochs
for epoch in range(30):

    # Train for one epoch
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

    # Print training and testing results
    print(
        f" Epoch {epoch}: "
        f"Train RMSE={rmse_train:.3f}, "
        f"Test RMSE={rmse_test:.3f}, "
        f"Score={score_test:.3f}"
    )

    # Save the best model
    if rmse_test < min_rmse:

        # Update best RMSE
        min_rmse = rmse_test

        # Update best score
        min_score = score_test

        # Ensure save directory exists
        os.makedirs("trained_model", exist_ok=True)

        # Save model parameters
        torch.save(
            model.state_dict(),
            f"E:/trained_model/rmse{rmse_test:.3f}_score{score_test:.3f}.pth"
        )
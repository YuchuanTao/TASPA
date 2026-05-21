import os
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

from model import TASPA
from data_process import CMAPSS


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

        # Iterate through test data
        for x, oc, y in dataloader:

            # Move data to device
            x, oc, y = x.to(device), oc.to(device), y.to(device)

            # Forward pass
            y_pred = model(x, oc)

            # Compute MSE loss
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

# Dataset selection (FD001, FD002, etc.)
dataset_class = "FD001"

# Maximum RUL clipping value
max_RUL = 125

# Sliding window length
time_window = 30

# K-fold validation setting
K_fold = 10

# Initialize CMAPSS dataset
cmapss = CMAPSS(dataset_class, max_RUL, time_window)

# Load training samples
train_samples, train_OC, train_RUL = cmapss.get_full_train_samples()

# Load test samples
test_samples, test_OC, test_RUL = cmapss.get_test_samples()

# Build test dataloader (full batch inference)
test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=len(test_samples),
    shuffle=False
)


# =======================
# Initialize model
# =======================

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


# =======================
# Load trained models
# =======================

# Model directory
model_dir = "trained_model/FD001"

# List all model files
model_files = [f for f in os.listdir(model_dir)]

# Store evaluation results
all_RMSE_list = []
all_Score_list = []


# =======================
# Evaluate all models
# =======================

# Loss function (not used in inference, but kept for consistency)
criterion = nn.MSELoss()

for model_file in model_files:

    # Construct full model path
    model_path = os.path.join(model_dir, model_file)

    # Load model weights
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # Set evaluation mode
    model.eval()

    # Print current model
    print(f"predicting：{model_file}")

    # Evaluate model
    rmse_test, score_test = test_one_epoch(
        model,
        test_loader,
        criterion
    )

    # Store results
    all_RMSE_list.append(rmse_test)
    all_Score_list.append(score_test)


# =======================
# Compute statistics
# =======================

RMSE_mean = np.mean(all_RMSE_list)
RMSE_std = np.std(all_RMSE_list)

Score_mean = np.mean(all_Score_list)
Score_std = np.std(all_Score_list)


# Print results
print(f"RMSE mean={RMSE_mean:.2f}, RMSE std={RMSE_std:.2f}")
print(f"Score mean={Score_mean:.2f}, Score std={Score_std:.2f}")
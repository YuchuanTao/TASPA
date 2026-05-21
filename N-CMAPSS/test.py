import os
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

from model import TASPA
from data_process import N_CMAPSS


def test_one_epoch(model, dataloader, loss_func):

    # Set model to evaluation mode
    model.eval()

    # Sum of squared errors
    SE = 0

    # Score function accumulator
    Score = 0

    with torch.no_grad():

        # Iterate through all test batches
        for x, oc, y in dataloader:

            # Move data to device
            x, oc, y = x.to(device), oc.to(device), y.to(device)

            # Forward prediction
            y_pred = model(x, oc)

            # Compute mean squared error loss
            loss = loss_func(y_pred, y)

            # Accumulate squared error
            SE += loss.item() * x.size(0)

            # Prediction error
            error = y_pred - y

            # Compute custom score function
            Score += torch.sum(
                torch.where(
                    error < 0,
                    torch.exp(-error / 13) - 1,
                    torch.exp(error / 10) - 1
                )
            ).item()

    # Compute RMSE
    RMSE = np.sqrt(SE / len(dataloader.dataset))

    # Average score
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
# Load test dataset
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

# Build test dataloader
test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=batch_size,
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
# Automatically find all model files
# =======================

# Directory containing trained models
model_dir = "./trained_model/"

# Get all files in model directory
model_files = [f for f in os.listdir(model_dir)]

# Store RMSE results
all_RMSE_list = []

# Store score results
all_Score_list = []


# =======================
# Load and evaluate models one by one
# =======================

# Mean squared error loss
criterion = nn.MSELoss()

for model_file in model_files:

    # Construct model path
    model_path = os.path.join(
        model_dir,
        "rmse5.304_score0.539.pth"
    )

    # Load model parameters
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # Set model to evaluation mode
    model.eval()

    # Print current model name
    print(f"predicting：{model_file}")

    # Evaluate model on test set
    rmse_test, score_test = test_one_epoch(
        model,
        test_loader,
        criterion
    )


# Print final RMSE
print(f"RMSE ={rmse_test:.2f}")

# Print final score
print(f"Scorea={score_test:.2f}")
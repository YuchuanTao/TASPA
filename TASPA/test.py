import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from matplotlib import pyplot as plt
from model import TASPA
from data_reprocess import CMAPSS  # CMAPSS class should support data_dir parameter

# =======================
# Hyperparameters
# =======================
dataset_class = "FD004"
max_RUL = 125
time_window = 60
K_fold = 10
data_dir = "./C-MAPSS_data"          # relative path to CMAPSS data
model_dir = "./trained_model"        # relative path to saved model
os.makedirs(model_dir, exist_ok=True)

TEM_kernel_size = 3
TEM_num_blocks = 2
attention_d_model = 32
attention_num_heads = 4
attention_dim_ff = 256
attention_num_layers = 4
ECA_kernel_size = 3
fused_hidden = 128
dropout = 0.1
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Load test dataset
# =======================
cmapss = CMAPSS(dataset_class, max_RUL, time_window, fold=1, K_fold=K_fold)
train_samples, train_OC, train_RUL = cmapss.get_full_train_samples()
test_samples, test_OC, test_RUL = cmapss.get_test_samples()

test_loader = Data.DataLoader(
    Data.TensorDataset(test_samples, test_OC, test_RUL),
    batch_size=len(test_samples),
    shuffle=False
)

# =======================
# Initialize model
# =======================
model = TASPA(
    TEM_kernel_size, TEM_num_blocks,
    time_window, 14,
    attention_d_model, attention_num_heads,
    attention_dim_ff, attention_num_layers,
    ECA_kernel_size, fused_hidden, dropout
).to(device)

# =======================
# Load trained weights
# =======================
model_path = os.path.join(model_dir, "TASPA_FD004.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =======================
# Inference
# =======================
all_predictions = []

with torch.no_grad():
    for batch_samples, batch_OC, batch_RUL in test_loader:
        batch_samples = batch_samples.to(device)
        batch_OC = batch_OC.to(device)

        outputs = model(batch_samples, batch_OC)
        outputs_numpy = outputs.detach().cpu().numpy()
        all_predictions.append(outputs_numpy)

# Convert list of batch predictions to a single numpy array
all_predictions = np.concatenate(all_predictions, axis=0)  # [N,1] -> still 2D

# Convert to 1D
pred_RUL = all_predictions.squeeze()  # [N]

# Convert true RUL to 1D numpy array
true_RUL = test_RUL.detach().cpu().numpy() if torch.is_tensor(test_RUL) else test_RUL
true_RUL = np.array(true_RUL).squeeze()

# Sort by true RUL
sorted_indices = np.argsort(true_RUL)
true_sorted = true_RUL[sorted_indices]
pred_sorted = pred_RUL[sorted_indices]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(true_sorted, label="True RUL", color="blue", linestyle="-")
plt.plot(pred_sorted, label="Predicted RUL", color="red", linestyle="--")
plt.xlabel("Samples (sorted by True RUL)")
plt.ylabel("RUL")
plt.title(f"True vs Predicted RUL (sorted) - {dataset_class}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

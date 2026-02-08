import os
import numpy as np
import torch
class CMAPSS:
    """
    CMAPSS dataset loader and window sampler for RUL prediction.

    Args:
        dataset_class (str): Dataset type, e.g., "FD001"
        max_RUL (int): Maximum RUL for clipping
        time_window_len (int): Sliding window length
        fold (int): Which fold is validation (1-based)
        K_fold (int): Total folds for cross-validation
    """
    def __init__(self, dataset_class, max_RUL, time_window_len, fold, K_fold):
        super().__init__()

        # Load raw numpy C-MAPSS_data

        data_dir = "C-MAPSS_data"
        self.train_data = np.load(os.path.join(data_dir, f"train_{dataset_class}.npy"))
        self.test_data = np.load(os.path.join(data_dir, f"test_{dataset_class}.npy"))
        self.test_RUL = np.load(os.path.join(data_dir, f"RUL_{dataset_class}.npy")).reshape(-1)


        # Select valid sensors
        self.valid_sensor_idx = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
        self.num_sensors = len(self.valid_sensor_idx)

        # Dataset size
        self.num_train_points = len(self.train_data)
        self.num_test_points = len(self.test_data)

        # Operation condition dimension
        self.oc_dim = 3

        self.max_RUL = max_RUL
        self.time_window_len = time_window_len

        # Cross-validation fold
        self.fold = fold
        self.K_fold = K_fold

        # Statistics for normalization (train)
        self.sensor_mean = np.zeros(self.num_sensors)
        self.sensor_std = np.zeros(self.num_sensors)
        self.oc_mean = np.zeros(self.oc_dim)
        self.oc_std = np.zeros(self.oc_dim)

        # Statistics for test set
        self.sensor_mean_test = np.zeros(self.num_sensors)
        self.sensor_std_test = np.zeros(self.num_sensors)
        self.oc_mean_test = np.zeros(self.oc_dim)
        self.oc_std_test = np.zeros(self.oc_dim)

    # --------------------------
    # Train sample generator
    # --------------------------
    def get_train_samples(self):
        """Return sliding window samples, OC, and RUL from the training set (excluding validation fold)."""
        data = self.train_data.copy()
        engine_ids = data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        # Exclude validation fold
        train_engines = unique_engines[
            (unique_engines <= (self.fold - 1) * (len(unique_engines) // self.K_fold)) |
            (unique_engines > self.fold * (len(unique_engines) // self.K_fold))
        ]
        train_data = data[np.isin(engine_ids, train_engines)]
        sensors = train_data[:, self.valid_sensor_idx].copy()
        oc = train_data[:, 2:5].copy()

        # Normalize sensors
        for i in range(self.num_sensors):
            mean = np.mean(sensors[:, i])
            std = np.std(sensors[:, i])
            self.sensor_mean[i] = mean
            self.sensor_std[i] = std
            sensors[:, i] = 0.0 if std == 0 else (sensors[:, i] - mean) / std

        # Normalize OC
        for i in range(self.oc_dim):
            mean = np.mean(oc[:, i])
            std = np.std(oc[:, i])
            self.oc_mean[i] = mean
            self.oc_std[i] = std
            oc[:, i] = 0.0 if std == 0 else (oc[:, i] - mean) / std

        # Create sliding windows
        X, X_OC, y = [], [], []
        engine_ids_train = engine_ids[np.isin(engine_ids, train_engines)]
        for engine in train_engines:
            idx = np.where(engine_ids_train == engine)[0]
            engine_sensors = sensors[idx]
            engine_oc = oc[idx]
            seq_len = len(idx)

            # Linear RUL decay with max_RUL clip
            engine_rul = np.clip(np.arange(seq_len, 0, -1) - 1, 0, self.max_RUL)

            for start in range(seq_len - self.time_window_len + 1):
                end = start + self.time_window_len
                X.append(engine_sensors[start:end])
                X_OC.append(engine_oc[start:end])
                y.append(engine_rul[end - 1])

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(X_OC), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32)
        )

    # --------------------------
    # Validation sample generator
    # --------------------------
    def get_validate_samples(self):
        """Return sliding window samples, OC, and RUL from the validation fold."""
        data = self.train_data.copy()
        engine_ids = data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        val_engines = unique_engines[
            (unique_engines > (self.fold - 1) * (len(unique_engines) // self.K_fold)) &
            (unique_engines <= self.fold * (len(unique_engines) // self.K_fold))
        ]
        val_data = data[np.isin(engine_ids, val_engines)]
        sensors = val_data[:, self.valid_sensor_idx].copy()
        oc = val_data[:, 2:5].copy()

        # Normalize with training statistics
        for i in range(self.num_sensors):
            std = self.sensor_std[i]
            sensors[:, i] = 0.0 if std == 0 else (sensors[:, i] - self.sensor_mean[i]) / std
        for i in range(self.oc_dim):
            std = self.oc_std[i]
            oc[:, i] = 0.0 if std == 0 else (oc[:, i] - self.oc_mean[i]) / std

        X, X_OC, y = [], [], []
        engine_ids_val = engine_ids[np.isin(engine_ids, val_engines)]
        for engine in val_engines:
            idx = np.where(engine_ids_val == engine)[0]
            engine_sensors = sensors[idx]
            engine_oc = oc[idx]
            seq_len = len(idx)
            engine_rul = np.clip(np.arange(seq_len, 0, -1) - 1, 0, self.max_RUL)

            for start in range(seq_len - self.time_window_len + 1):
                end = start + self.time_window_len
                X.append(engine_sensors[start:end])
                X_OC.append(engine_oc[start:end])
                y.append(engine_rul[end - 1])

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(X_OC), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32)
        )

    # --------------------------
    # Test sample generator
    # --------------------------
    def get_test_samples(self):
        """Return last-window samples for each engine in the test set."""
        sensors = self.test_data[:, self.valid_sensor_idx].copy()
        oc = self.test_data[:, 2:5].copy()
        RUL = self.test_RUL.copy()

        # Normalize with test statistics
        for i in range(self.num_sensors):
            std = self.sensor_std_test[i]
            sensors[:, i] = 0.0 if std == 0 else (sensors[:, i] - self.sensor_mean_test[i]) / std
        for i in range(self.oc_dim):
            std = self.oc_std_test[i]
            oc[:, i] = 0.0 if std == 0 else (oc[:, i] - self.oc_mean_test[i]) / std

        X, X_OC, y = [], [], []
        engine_ids = self.test_data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        for engine in unique_engines:
            idx = np.where(engine_ids == engine)[0]
            seq_len = len(idx)
            if seq_len < self.time_window_len:
                continue
            X.append(sensors[idx][-self.time_window_len:])
            X_OC.append(oc[idx][-self.time_window_len:])
            y.append(min(RUL[engine - 1], self.max_RUL))

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(X_OC), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32)
        )

    # --------------------------
    # Full train set (for test/inference)
    # --------------------------
    def get_full_train_samples(self):
        """Return sliding window samples for the full training set, no fold splitting."""
        data = self.train_data.copy()
        engine_ids = data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        sensors = data[:, self.valid_sensor_idx].copy()
        oc = data[:, 2:5].copy()

        # Compute test normalization statistics
        for i in range(self.num_sensors):
            mean = np.mean(sensors[:, i])
            std = np.std(sensors[:, i])
            self.sensor_mean_test[i] = mean
            self.sensor_std_test[i] = std
            sensors[:, i] = 0.0 if std == 0 else (sensors[:, i] - mean) / std
        for i in range(self.oc_dim):
            mean = np.mean(oc[:, i])
            std = np.std(oc[:, i])
            self.oc_mean_test[i] = mean
            self.oc_std_test[i] = std
            oc[:, i] = 0.0 if std == 0 else (oc[:, i] - mean) / std

        X, X_OC, y = [], [], []
        for engine in unique_engines:
            idx = np.where(engine_ids == engine)[0]
            engine_sensors = sensors[idx]
            engine_oc = oc[idx]
            seq_len = len(idx)
            engine_rul = np.clip(np.arange(seq_len, 0, -1) - 1, 0, self.max_RUL)

            for start in range(seq_len - self.time_window_len + 1):
                end = start + self.time_window_len
                X.append(engine_sensors[start:end])
                X_OC.append(engine_oc[start:end])
                y.append(engine_rul[end - 1])

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(X_OC), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32)
        )

import h5py
import numpy as np
import pandas as pd
import torch


class N_CMAPSS:
    """
    N-C-MAPSS dataset processing class
    """

    def __init__(self, file_path, sample_step, time_window):

        # Sampling interval
        self.sample_step = sample_step

        # Sliding time window length
        self.time_window = time_window

        # Read HDF5 dataset
        with h5py.File(file_path, 'r') as hdf:

            # Development (training) set
            self.A_dev = np.array(hdf['A_dev'])
            self.T_dev = np.array(hdf['T_dev'])
            self.W_dev = np.array(hdf['W_dev'])
            self.X_s_dev = np.array(hdf['X_s_dev'])
            self.X_v_dev = np.array(hdf['X_v_dev'])
            self.Y_dev = np.array(hdf['Y_dev'])

            # Test set
            self.A_test = np.array(hdf['A_test'])
            self.T_test = np.array(hdf['T_test'])
            self.W_test = np.array(hdf['W_test'])
            self.X_s_test = np.array(hdf['X_s_test'])
            self.X_v_test = np.array(hdf['X_v_test'])
            self.Y_test = np.array(hdf['Y_test'])

            # Decode variable names from bytes to strings
            self.A_cols = [c.decode('utf-8') for c in hdf['A_var']]
            self.T_cols = [c.decode('utf-8') for c in hdf['T_var']]
            self.W_cols = [c.decode('utf-8') for c in hdf['W_var']]
            self.X_s_cols = [c.decode('utf-8') for c in hdf['X_s_var']]
            self.X_v_cols = [c.decode('utf-8') for c in hdf['X_v_var']]

        # Original training dataframe without preprocessing
        self.origin_data_frame_in_train_set = pd.DataFrame(
            np.hstack([
                self.A_dev,
                self.T_dev,
                self.W_dev,
                self.X_s_dev,
                self.X_v_dev,
                self.Y_dev
            ]),
            columns=self.A_cols +
                    self.T_cols +
                    self.W_cols +
                    self.X_s_cols +
                    self.X_v_cols +
                    ['RUL']
        )

        # Original training numpy array
        self.origin_data_in_train_set = np.hstack([
            self.A_dev,
            self.T_dev,
            self.W_dev,
            self.X_s_dev,
            self.X_v_dev,
            self.Y_dev
        ])

        # Original test dataframe without preprocessing
        self.origin_data_frame_in_test_set = pd.DataFrame(
            np.hstack([
                self.A_test,
                self.T_test,
                self.W_test,
                self.X_s_test,
                self.X_v_test,
                self.Y_test
            ]),
            columns=self.A_cols +
                    self.T_cols +
                    self.W_cols +
                    self.X_s_cols +
                    self.X_v_cols +
                    ['RUL']
        )

        # Original test numpy array
        self.origin_data_in_test_set = np.hstack([
            self.A_test,
            self.T_test,
            self.W_test,
            self.X_s_test,
            self.X_v_test,
            self.Y_test
        ])

        # Selected sensor signals
        # self.selected_signals = [
        #     'alt', 'Mach', 'TRA', 'T2',
        #     'Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50',
        #     'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50'
        # ]

        self.selected_signals = [
            'Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50',
            'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50'
        ]

        # Selected operating condition (OC) variables
        self.selected_OC = [
            'alt', 'Mach', 'TRA', 'T2',
        ]

        # Sensor feature column names
        self.feature_cols = [
            col for col in self.selected_signals
            if col in self.origin_data_frame_in_train_set.columns
        ]

        # Indices of selected sensor features
        self.feature_indices = [
            self.origin_data_frame_in_train_set.columns.get_loc(col)
            for col in self.feature_cols
        ]

        # Operating condition column names
        self.OC_cols = [
            col for col in self.selected_signals
            if col in self.origin_data_frame_in_train_set.columns
        ]

        # Indices of operating condition features
        self.OC_indices = [
            self.origin_data_frame_in_train_set.columns.get_loc(col)
            for col in self.feature_cols
        ]

        # Number of selected sensors
        self.valid_sensor_number = len(self.feature_indices)

        # Mean values of sensors for test normalization
        self.valid_sensor_mean_for_test = np.zeros(self.valid_sensor_number)

        # Standard deviation values of sensors for test normalization
        self.valid_sensor_std_for_test = np.zeros(self.valid_sensor_number)

        # Mean values of OC variables for test normalization
        self.OC_mean_for_test = np.zeros(4)

        # Standard deviation values of OC variables for test normalization
        self.OC_std_for_test = np.zeros(4)

    def get_train_data(self):

        # Copy original training data
        origin_data = self.origin_data_in_train_set.copy()

        # Downsample data
        origin_data = origin_data[::self.sample_step, :]

        # Extract RUL labels
        RUL_data = origin_data[:, -1]

        # Extract selected sensor features
        sensor_data = origin_data[:, self.feature_indices].copy()

        # Extract operating condition features
        OC_data = origin_data[:, self.OC_indices].copy()

        # Engine index for each flight cycle
        engine_fly_time_index = origin_data[:, 0].astype(int)

        # Unique engine IDs
        engine_index = np.unique(engine_fly_time_index)

        # Z-score normalization for sensor data
        for i in range(self.valid_sensor_number):

            mean = np.mean(sensor_data[:, i])
            std = np.std(sensor_data[:, i])

            if std == 0:
                sensor_data[:, i] = 0.0
            else:
                sensor_data[:, i] = (sensor_data[:, i] - mean) / std

            # Save normalization parameters
            self.valid_sensor_mean_for_test[i] = mean
            self.valid_sensor_std_for_test[i] = std

        # Z-score normalization for operating condition data
        for i in range(4):

            mean = np.mean(OC_data[:, i])
            std = np.std(OC_data[:, i])

            if std == 0:
                OC_data[:, i] = 0.0
            else:
                OC_data[:, i] = (OC_data[:, i] - mean) / std

            # Save normalization parameters
            self.OC_mean_for_test[i] = mean
            self.OC_std_for_test[i] = std

        # Containers for sliding window samples
        Time_Window_Samples = []
        Time_Window_RUL = []
        Time_Window_OC = []

        # Iterate over all engines
        for engine_id in engine_index:

            # Get indices of all cycles for current engine
            fly_time = np.where(engine_fly_time_index == engine_id)[0]

            # Sensor sequence of current engine
            engine_sensor_data = sensor_data[fly_time, :]

            # OC sequence of current engine
            engine_OC_data = OC_data[fly_time, :]

            # Number of cycles
            fly_time_len = len(fly_time)

            # RUL sequence
            engine_rul = RUL_data[fly_time]

            # Construct sliding time window samples
            for start in range(fly_time_len - self.time_window + 1):

                end = start + self.time_window

                # Sensor window: [T, feature_dim]
                Time_Window_Samples.append(
                    engine_sensor_data[start:end, :]
                )

                # OC window
                Time_Window_OC.append(
                    engine_OC_data[start:end, :]
                )

                # RUL label of the last timestep
                Time_Window_RUL.append(
                    engine_rul[end - 1]
                )

        # Convert lists to numpy arrays
        Time_Window_Samples = np.array(Time_Window_Samples)
        Time_Window_OC = np.array(Time_Window_OC)
        Time_Window_RUL = np.array(Time_Window_RUL)

        # Convert numpy arrays to PyTorch tensors
        Time_Window_Samples = torch.tensor(
            Time_Window_Samples,
            dtype=torch.float32
        )

        Time_Window_OC = torch.tensor(
            Time_Window_OC,
            dtype=torch.float32
        )

        Time_Window_RUL = torch.tensor(
            Time_Window_RUL,
            dtype=torch.float32
        )

        return Time_Window_Samples, Time_Window_OC, Time_Window_RUL

    def get_test_data(self):

        # Copy original test data
        origin_data = self.origin_data_in_test_set.copy()

        # Downsample data
        origin_data = origin_data[::self.sample_step, :]

        # Extract RUL labels
        RUL_data = origin_data[:, -1]

        # Extract operating condition features
        OC_data = origin_data[:, self.OC_indices].copy()

        # Extract selected sensor features
        sensor_data = origin_data[:, self.feature_indices].copy()

        # Engine index for each flight cycle
        engine_fly_time_index = origin_data[:, 0].astype(int)

        # Unique engine IDs
        engine_index = np.unique(engine_fly_time_index)

        # Normalize sensor data using training statistics
        for i in range(self.valid_sensor_number):

            mean = self.valid_sensor_mean_for_test[i]
            std = self.valid_sensor_std_for_test[i]

            if std == 0:
                sensor_data[:, i] = 0.0
            else:
                sensor_data[:, i] = (sensor_data[:, i] - mean) / std

        # Normalize operating condition data using training statistics
        for i in range(4):

            mean = self.OC_mean_for_test[i]
            std = self.OC_std_for_test[i]

            if std == 0:
                OC_data[:, i] = 0.0
            else:
                OC_data[:, i] = (OC_data[:, i] - mean) / std

        # Containers for sliding window samples
        Time_Window_Samples = []
        Time_Window_RUL = []
        Time_Window_OC = []

        # Iterate over all engines
        for engine_id in engine_index:

            # Get indices of all cycles for current engine
            fly_time = np.where(engine_fly_time_index == engine_id)[0]

            # Sensor sequence of current engine
            engine_sensor_data = sensor_data[fly_time, :]

            # OC sequence of current engine
            engine_OC_data = OC_data[fly_time, :]

            # Number of cycles
            fly_time_len = len(fly_time)

            # RUL sequence
            engine_rul = RUL_data[fly_time]

            # Construct sliding time window samples
            for start in range(fly_time_len - self.time_window + 1):

                end = start + self.time_window

                # Sensor window: [T, feature_dim]
                Time_Window_Samples.append(
                    engine_sensor_data[start:end, :]
                )

                # OC window
                Time_Window_OC.append(
                    engine_OC_data[start:end, :]
                )

                # RUL label of the last timestep
                Time_Window_RUL.append(
                    engine_rul[end - 1]
                )

        # Convert lists to numpy arrays
        Time_Window_Samples = np.array(Time_Window_Samples)
        Time_Window_OC = np.array(Time_Window_OC)
        Time_Window_RUL = np.array(Time_Window_RUL)

        # Convert numpy arrays to PyTorch tensors
        Time_Window_Samples = torch.tensor(
            Time_Window_Samples,
            dtype=torch.float32
        )

        Time_Window_OC = torch.tensor(
            Time_Window_OC,
            dtype=torch.float32
        )

        Time_Window_RUL = torch.tensor(
            Time_Window_RUL,
            dtype=torch.float32
        )

        return Time_Window_Samples, Time_Window_OC, Time_Window_RUL
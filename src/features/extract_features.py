# import numpy as np

# def flatten_window(window):
#     return np.array(window).flatten()

import numpy as np

def extract_features(data_window):
    """
    Extract statistical features from a single IMU data window.

    Parameters:
    - data_window (np.array): shape (N, F) where N is samples and F is IMU features (e.g., acc_x, acc_y, ...)

    Returns:
    - np.array: 1D feature vector
    """
    features = []

    # لو عندك 6 features مثلاً: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    for i in range(data_window.shape[1]):
        signal = data_window[:, i]

        # Features: mean, std, max, min, median, energy
        mean = np.mean(signal)
        std = np.std(signal)
        max_val = np.max(signal)
        min_val = np.min(signal)
        median = np.median(signal)
        energy = np.sum(signal ** 2) / len(signal)

        features.extend([mean, std, max_val, min_val, median, energy])

    return np.array(features)

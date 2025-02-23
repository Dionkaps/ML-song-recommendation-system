import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal


def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
    data = data.astype(float)
    if data.ndim == 1:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = np.clip(data, lower_bound, upper_bound)
    elif data.ndim == 2:
        for i in range(data.shape[0]):
            row = data[i]
            Q1 = np.percentile(row, 25)
            Q3 = np.percentile(row, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[i] = np.clip(row, lower_bound, upper_bound)
    else:
        raise ValueError(
            "Only 1D and 2D arrays are supported for outlier removal.")
    return data


def scale_feature(data: np.ndarray,
                  scaler_type: Literal["standard", "minmax"] = "standard") -> np.ndarray:
    data = data.astype(float)
    transposed = False
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim == 2:
        data = data.T
        transposed = True
    else:
        raise ValueError("Only 1D and 2D arrays are supported for scaling.")
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'.")
    data_scaled = scaler.fit_transform(data)
    if transposed:
        data_scaled = data_scaled.T
    else:
        data_scaled = data_scaled.flatten()
    return data_scaled


def apply_log_transform(data: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    data = data.astype(float)
    return np.log(data + 1 + offset)


def process_feature_file(file_path: str, scaler_type: str = "standard") -> None:
    print(f"\nProcessing file: {file_path}")
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    if "zero_crossing_rate" in file_path or "spectral_flatness" in file_path:
        data = apply_log_transform(data)
        print("  - Applied logarithmic transformation.")
    data = remove_outliers_iqr(data)
    print("  - Removed outliers using IQR.")
    data = scale_feature(data, scaler_type=scaler_type)
    print(f"  - Scaled data using the {scaler_type.capitalize()} scaler.")
    base_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(base_dir)
    processed_dir = os.path.join(parent_dir, "processed_results")
    os.makedirs(processed_dir, exist_ok=True)
    processed_file_name = os.path.basename(
        file_path).replace(".npy", "_processed.npy")
    processed_file_path = os.path.join(processed_dir, processed_file_name)
    np.save(processed_file_path, data)
    print(f"  - Saved processed feature to: {processed_file_path}")


def process_all_features(results_dir: str = "results", scaler_type: str = "standard") -> None:
    feature_files = glob.glob(os.path.join(results_dir, "*.npy"))
    feature_files = [f for f in feature_files if "_processed.npy" not in f]
    if not feature_files:
        print(f"No unprocessed feature files found in '{results_dir}'.")
        return
    for file_path in feature_files:
        process_feature_file(file_path, scaler_type=scaler_type)


if __name__ == "__main__":
    scaler_choice = "standard"
    process_all_features(results_dir="results", scaler_type=scaler_choice)

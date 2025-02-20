import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal


# ------------------------------
# Helper Functions
# ------------------------------
def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
    """
    Removes outliers from a 1D or 2D NumPy array using the IQR method.
    For a 1D array, the clipping is applied to the entire array.
    For a 2D array, the clipping is applied row by row.

    Parameters
    ----------
    data : np.ndarray
        Input array (1D or 2D).

    Returns
    -------
    np.ndarray
        The array with outliers clipped.
    """
    data = data.astype(float)  # Ensure float

    if data.ndim == 1:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = np.clip(data, lower_bound, upper_bound)
    elif data.ndim == 2:
        # Clip outliers row by row
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
    """
    Scales a 1D or 2D NumPy array using either StandardScaler or MinMaxScaler.
    - 1D: treat the entire array as one feature.
    - 2D: by default, each row is treated as one feature (transposing before fitting).

    Parameters
    ----------
    data : np.ndarray
        Input array (1D or 2D).
    scaler_type : {'standard', 'minmax'}, optional
        The type of scaler to use. Defaults to 'standard'.

    Returns
    -------
    np.ndarray
        The scaled array with the original shape preserved.
    """
    data = data.astype(float)  # Ensure float
    transposed = False

    # 1D case -> shape to (n_samples, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # 2D case -> transpose so scikit-learn sees each row as a single feature
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

    # Restore original shape
    if transposed:
        data_scaled = data_scaled.T
    else:
        # Flatten back to 1D if the original input was 1D
        data_scaled = data_scaled.flatten()

    return data_scaled


def apply_log_transform(data: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    """
    Applies a log transform to the data. Uses log(1 + x + offset) to avoid log(0).

    Parameters
    ----------
    data : np.ndarray
        Input array (1D or 2D).
    offset : float, optional
        Small value added to avoid log(0). Defaults to 1e-6.

    Returns
    -------
    np.ndarray
        The log-transformed array.
    """
    data = data.astype(float)
    # We add `1 + offset` if we want strictly positive input to np.log
    return np.log(data + 1 + offset)


# ------------------------------
# Processing Functions
# ------------------------------
def process_feature_file(file_path: str, scaler_type: str = "standard") -> None:
    """
    Loads a .npy feature file, optionally applies log transform (based on filename),
    removes outliers, scales the data, and saves the result to a 'processed_results' directory.
    """
    print(f"\nProcessing file: {file_path}")
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # Apply log transform for certain feature types
    if "zero_crossing_rate" in file_path or "spectral_flatness" in file_path:
        data = apply_log_transform(data)
        print("  - Applied logarithmic transformation.")

    # Remove outliers
    data = remove_outliers_iqr(data)
    print("  - Removed outliers using IQR.")

    # Scale the data
    data = scale_feature(data, scaler_type=scaler_type)
    print(f"  - Scaled data using the {scaler_type.capitalize()} scaler.")

    # Prepare the output directory
    base_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(base_dir)
    processed_dir = os.path.join(parent_dir, "processed_results")
    os.makedirs(processed_dir, exist_ok=True)

    # Save processed data
    processed_file_name = os.path.basename(
        file_path).replace(".npy", "_processed.npy")
    processed_file_path = os.path.join(processed_dir, processed_file_name)
    np.save(processed_file_path, data)
    print(f"  - Saved processed feature to: {processed_file_path}")


def process_all_features(results_dir: str = "results", scaler_type: str = "standard") -> None:
    """
    Processes all .npy files in the given directory (except those already processed)
    by applying log transform (if needed), outlier removal, and scaling.
    """
    feature_files = glob.glob(os.path.join(results_dir, "*.npy"))
    feature_files = [f for f in feature_files if "_processed.npy" not in f]

    if not feature_files:
        print(f"No unprocessed feature files found in '{results_dir}'.")
        return

    for file_path in feature_files:
        process_feature_file(file_path, scaler_type=scaler_type)


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    scaler_choice = "standard"
    process_all_features(results_dir="results", scaler_type=scaler_choice)

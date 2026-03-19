import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import feature_vars as fv


FEATURE_KEYS: Tuple[str, ...] = tuple(fv.AUDIO_FEATURE_KEYS)
_PARSE_ORDER: Tuple[str, ...] = tuple(
    sorted(FEATURE_KEYS, key=len, reverse=True)
)


@dataclass
class FeatureBundleValidationResult:
    base_name: str
    status: str
    missing_keys: Tuple[str, ...]
    invalid_keys: Tuple[str, ...]
    present_keys: Tuple[str, ...]
    issue_details: Dict[str, str]

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> Dict[str, str]:
        return {
            "BaseName": self.base_name,
            "Status": self.status,
            "MissingKeys": ",".join(self.missing_keys),
            "InvalidKeys": ",".join(self.invalid_keys),
            "PresentKeys": ",".join(self.present_keys),
            "IssueDetails": json.dumps(self.issue_details, sort_keys=True),
        }


def get_feature_output_paths(
    base_filename: str,
    results_dir: str,
    feature_keys: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    selected_keys = tuple(feature_keys or FEATURE_KEYS)
    return {
        key: os.path.join(results_dir, f"{base_filename}_{key}.npy")
        for key in selected_keys
    }


def split_feature_filename(filename: str) -> Optional[Tuple[str, str]]:
    for key in _PARSE_ORDER:
        suffix = f"_{key}.npy"
        if filename.endswith(suffix):
            return filename[: -len(suffix)], key
    return None


def collect_feature_bundle_inventory(
    results_dir: str,
    feature_keys: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    selected = set(feature_keys or FEATURE_KEYS)
    inventory: Dict[str, set] = {}

    if not os.path.isdir(results_dir):
        return {}

    for filename in os.listdir(results_dir):
        parsed = split_feature_filename(filename)
        if parsed is None:
            continue
        base_name, key = parsed
        if key not in selected:
            continue
        inventory.setdefault(base_name, set()).add(key)

    return {
        base_name: sorted(keys)
        for base_name, keys in inventory.items()
    }


def audio_library_basenames(audio_dir: str) -> List[str]:
    if not audio_dir or not os.path.isdir(audio_dir):
        return []

    basenames = set()
    for extension in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
        basenames.update(Path(path).stem for path in Path(audio_dir).glob(extension))
    return sorted(basenames)


def validate_feature_array(
    key: str,
    arr: np.ndarray,
    n_mfcc: int = fv.n_mfcc,
    n_chroma: int = fv.n_chroma,
) -> None:
    if arr.size == 0:
        raise ValueError(f"{key} array is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} contains NaN or Inf values")

    expected_dims = {
        "mfcc": n_mfcc,
        "delta_mfcc": n_mfcc,
        "delta2_mfcc": n_mfcc,
        "spectral_centroid": 1,
        "spectral_rolloff": 1,
        "spectral_flux": 1,
        "spectral_flatness": 1,
        "zero_crossing_rate": 1,
        "chroma": n_chroma,
    }

    if key == "beat_strength":
        flattened = np.asarray(arr).reshape(-1)
        if flattened.size != 4:
            raise ValueError(
                f"beat_strength must flatten to 4 scalars, got shape {arr.shape}"
            )
        return

    if arr.ndim != 2:
        raise ValueError(f"{key} must be 2D, got shape {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError(f"{key} has zero frames")

    expected_first_dim = expected_dims.get(key)
    if expected_first_dim is not None and arr.shape[0] != expected_first_dim:
        raise ValueError(
            f"{key} expected first dimension {expected_first_dim}, got {arr.shape[0]}"
        )


def load_validated_feature_bundle(
    base_name: str,
    results_dir: str,
    feature_keys: Optional[Sequence[str]] = None,
    n_mfcc: int = fv.n_mfcc,
    n_chroma: int = fv.n_chroma,
) -> Tuple[Optional[Dict[str, np.ndarray]], FeatureBundleValidationResult]:
    selected_keys = tuple(feature_keys or FEATURE_KEYS)
    paths = get_feature_output_paths(base_name, results_dir, selected_keys)
    arrays: Dict[str, np.ndarray] = {}
    missing_keys: List[str] = []
    invalid_keys: List[str] = []
    issue_details: Dict[str, str] = {}

    for key in selected_keys:
        path = paths[key]
        if not os.path.isfile(path):
            missing_keys.append(key)
            issue_details[key] = "missing"
            continue

        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:
            invalid_keys.append(key)
            issue_details[key] = f"load_error: {exc}"
            continue

        try:
            validate_feature_array(
                key,
                arr,
                n_mfcc=n_mfcc,
                n_chroma=n_chroma,
            )
        except ValueError as exc:
            invalid_keys.append(key)
            issue_details[key] = str(exc)
            continue

        arrays[key] = np.asarray(arr)

    if missing_keys:
        status = "incomplete"
    elif invalid_keys:
        status = "invalid"
    else:
        status = "ok"

    result = FeatureBundleValidationResult(
        base_name=base_name,
        status=status,
        missing_keys=tuple(sorted(missing_keys)),
        invalid_keys=tuple(sorted(invalid_keys)),
        present_keys=tuple(sorted(arrays)),
        issue_details=issue_details,
    )
    return (arrays if status == "ok" else None), result


def remove_feature_bundle(
    base_name: str,
    results_dir: str,
    feature_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    removed: List[str] = []
    for path in get_feature_output_paths(base_name, results_dir, feature_keys).values():
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return removed

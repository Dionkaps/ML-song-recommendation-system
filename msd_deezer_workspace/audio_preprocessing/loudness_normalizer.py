"""
Loudness Normalizer for Audio Preprocessing.

Implements ITU-R BS.1770 integrated loudness measurement and normalization
with a sample-peak safety ceiling.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pyloudnorm as pyln


logger = logging.getLogger(__name__)


class LoudnessNormalizer:
    """
    Normalizes audio loudness using ITU-R BS.1770 measurement.

    This class implements a two-stage loudness normalization:
    1. Measure integrated loudness using ITU-R BS.1770
    2. Apply gain to reach target LUFS
    3. Apply a sample-peak safety cap if peak exceeds threshold
    """

    def __init__(self, target_lufs: float = -14.0, max_true_peak: float = -1.0):
        if target_lufs > 0:
            raise ValueError("target_lufs should be negative (typically -14 to -23)")
        if max_true_peak > 0:
            raise ValueError("max_true_peak should be negative or zero")

        self.target_lufs = target_lufs
        self.max_true_peak = max_true_peak
        self._max_peak_linear = 10.0 ** (max_true_peak / 20.0)

    def measure_loudness(self, y: np.ndarray, sr: int) -> float:
        meter = pyln.Meter(sr)
        return meter.integrated_loudness(y)

    def measure_sample_peak(self, y: np.ndarray) -> float:
        peak_linear = np.max(np.abs(y))
        if peak_linear > 0:
            return 20 * np.log10(peak_linear)
        return -np.inf

    measure_true_peak = measure_sample_peak

    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        stats = {
            "original_lufs": None,
            "final_lufs": None,
            "original_peak_db": None,
            "final_peak_db": None,
            "gain_applied_db": 0.0,
            "actions": [],
        }

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        stats["original_lufs"] = loudness
        stats["original_peak_db"] = self.measure_sample_peak(y)

        if np.isinf(loudness):
            stats["actions"].append("skipped_silence")
            stats["final_lufs"] = loudness
            stats["final_peak_db"] = stats["original_peak_db"]
            return y, stats

        delta_lufs = self.target_lufs - loudness
        gain = 10.0 ** (delta_lufs / 20.0)
        y_normalized = y * gain

        peak = np.max(np.abs(y_normalized))
        if peak > self._max_peak_linear:
            limiter_gain = self._max_peak_linear / peak
            y_normalized = y_normalized * limiter_gain
            stats["actions"].append("peak_limited")
            gain = gain * limiter_gain

        stats["gain_applied_db"] = 20 * np.log10(gain)
        stats["final_lufs"] = meter.integrated_loudness(y_normalized)
        stats["final_peak_db"] = self.measure_sample_peak(y_normalized)
        stats["actions"].append("normalized")

        return y_normalized, stats

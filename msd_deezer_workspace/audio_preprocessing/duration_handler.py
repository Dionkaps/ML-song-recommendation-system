"""
Duration Handler for Audio Preprocessing.

This module handles audio duration validation and cropping to ensure
all audio files are exactly the target duration (default: 29 seconds).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


class DurationHandler:
    """
    Handles audio duration validation and cropping.

    This class ensures all audio files meet the target duration requirement:
    - Files shorter than target duration are flagged for removal
    - Files longer than target duration are cropped from the beginning
    - Files exactly at target duration pass through unchanged
    """

    def __init__(self, target_duration: float = 29.0, min_duration: float | None = None):
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        self.target_duration = target_duration
        self.min_duration = min_duration if min_duration is not None else target_duration

    def get_duration(self, y: np.ndarray, sr: int) -> float:
        return len(y) / sr

    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, str]:
        duration = self.get_duration(y, sr)

        if duration < self.min_duration:
            logger.debug("Audio too short: %.2fs < %.2fs", duration, self.min_duration)
            return y, "too_short"

        if duration > self.target_duration:
            target_samples = int(self.target_duration * sr)
            y_cropped = y[:target_samples]
            logger.debug("Cropped audio: %.2fs -> %.2fs", duration, self.target_duration)
            return y_cropped, "cropped"

        return y, "ok"

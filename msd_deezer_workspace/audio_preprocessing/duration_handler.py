"""
Duration Handler for Audio Preprocessing.

Validates that an audio excerpt is long enough and, when longer than the
target duration, crops a fixed-length window. The default `center` crop
location follows the MIR convention (Tzanetakis & Cook, 2002; MSD 30 s
middle-excerpt previews via 7digital, Bertin-Mahieux et al., 2011): the
middle of the track is the most representative window, avoiding intros /
outros that tend to be atypical (silence, fade-in, sparse instrumentation).

For Deezer previews (which are already middle excerpts by construction)
either `center` or `start` produces essentially the same window, so
`center` is a safe default that also generalises to full-length tracks.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


class DurationHandler:
    """Decide whether an audio signal meets the target duration and crop it."""

    VALID_CROP_LOCATIONS = ("center", "start")

    def __init__(
        self,
        target_duration: float = 29.0,
        min_duration: float | None = None,
        crop_location: str = "center",
    ):
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if crop_location not in self.VALID_CROP_LOCATIONS:
            raise ValueError(
                f"crop_location must be one of {self.VALID_CROP_LOCATIONS}, "
                f"got {crop_location!r}"
            )

        self.target_duration = target_duration
        self.min_duration = min_duration if min_duration is not None else target_duration
        self.crop_location = crop_location

    def get_duration(self, y: np.ndarray, sr: int) -> float:
        return len(y) / sr

    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, str]:
        duration = self.get_duration(y, sr)

        if duration < self.min_duration:
            logger.debug("Audio too short: %.2fs < %.2fs", duration, self.min_duration)
            return y, "too_short"

        if duration > self.target_duration:
            target_samples = int(self.target_duration * sr)
            if self.crop_location == "center":
                start = max(0, (len(y) - target_samples) // 2)
            else:  # "start"
                start = 0
            y_cropped = y[start:start + target_samples]
            logger.debug(
                "Cropped audio (%s): %.2fs -> %.2fs",
                self.crop_location, duration, self.target_duration,
            )
            return y_cropped, "cropped"

        return y, "ok"

"""
Duration Handler for Audio Preprocessing.

This module handles audio duration validation and cropping to ensure
all audio files are exactly the target duration (default: 29 seconds).

Part of the EBU R128 compliant audio preprocessing pipeline.
"""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class DurationHandler:
    """
    Handles audio duration validation and cropping.
    
    This class ensures all audio files meet the target duration requirement:
    - Files shorter than target duration are flagged for removal
    - Files longer than target duration are cropped (from the beginning)
    - Files exactly at target duration pass through unchanged
    
    Attributes:
        target_duration (float): Target duration in seconds
        min_duration (float): Minimum acceptable duration (default: target_duration)
    """
    
    def __init__(self, target_duration: float = 29.0, min_duration: float = None):
        """
        Initialize DurationHandler.
        
        Args:
            target_duration: Target duration in seconds (default: 29.0)
            min_duration: Minimum acceptable duration. If None, uses target_duration.
                          Set lower to allow padding short files instead of removing.
        
        Raises:
            ValueError: If target_duration <= 0
        """
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
            
        self.target_duration = target_duration
        self.min_duration = min_duration if min_duration is not None else target_duration

    def get_duration(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate duration of audio signal.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Duration in seconds
        """
        return len(y) / sr

    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, str]:
        """
        Check duration and crop if necessary.
        
        Args:
            y: Audio time series (numpy array)
            sr: Sampling rate in Hz
            
        Returns:
            Tuple of (processed_audio, action_taken)
            action_taken can be: 'ok', 'cropped', 'too_short'
            
        Note:
            Cropping is done from the start of the audio (keeps first N seconds).
            This is appropriate for music where intros are often representative.
        """
        duration = self.get_duration(y, sr)
        
        if duration < self.min_duration:
            logger.debug(f"Audio too short: {duration:.2f}s < {self.min_duration:.2f}s")
            return y, 'too_short'
        
        if duration > self.target_duration:
            target_samples = int(self.target_duration * sr)
            y_cropped = y[:target_samples]
            logger.debug(f"Cropped audio: {duration:.2f}s -> {self.target_duration:.2f}s")
            return y_cropped, 'cropped'
            
        return y, 'ok'

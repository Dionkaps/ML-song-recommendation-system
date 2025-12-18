"""
Loudness Normalizer for Audio Preprocessing.

Implements ITU-R BS.1770 integrated loudness measurement and normalization
with EBU R128 compliant True Peak limiting.

Standards Reference:
- ITU-R BS.1770-4: Algorithms to measure audio programme loudness
- EBU R128: Loudness normalisation and permitted maximum level of audio signals

Typical Target Values:
- Streaming platforms: -14 LUFS (Spotify, YouTube)
- Broadcast (EBU R128): -23 LUFS
- Apple Music: -16 LUFS
"""

import logging
import numpy as np
import pyloudnorm as pyln
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class LoudnessNormalizer:
    """
    Normalizes audio loudness using ITU-R BS.1770 measurement.
    
    This class implements a two-stage loudness normalization:
    1. Measure integrated loudness using ITU-R BS.1770
    2. Apply gain to reach target LUFS
    3. Apply True Peak limiting if peak exceeds threshold
    
    The True Peak limiter uses simple gain reduction (not a full limiter)
    to ensure peaks don't exceed the threshold. This may result in 
    slightly lower final loudness than target if limiting is applied.
    
    Attributes:
        target_lufs (float): Target integrated loudness in LUFS
        max_true_peak (float): Maximum true peak in dBTP
    """
    
    def __init__(self, target_lufs: float = -14.0, max_true_peak: float = -1.0):
        """
        Initialize LoudnessNormalizer.
        
        Args:
            target_lufs: Target Integrated Loudness in LUFS (default: -14.0)
                        Common values: -14 (streaming), -23 (broadcast), -16 (Apple)
            max_true_peak: Maximum True Peak in dBTP (default: -1.0)
                          EBU R128 recommends -1.0 dBTP for broadcast
        
        Raises:
            ValueError: If target_lufs > 0 or max_true_peak > 0
        """
        if target_lufs > 0:
            raise ValueError("target_lufs should be negative (typically -14 to -23)")
        if max_true_peak > 0:
            raise ValueError("max_true_peak should be negative or zero")
            
        self.target_lufs = target_lufs
        self.max_true_peak = max_true_peak
        self._max_peak_linear = 10.0 ** (max_true_peak / 20.0)

    def measure_loudness(self, y: np.ndarray, sr: int) -> float:
        """
        Measure integrated loudness of audio signal.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Integrated loudness in LUFS (may be -inf for silence)
        """
        meter = pyln.Meter(sr)
        return meter.integrated_loudness(y)

    def measure_true_peak(self, y: np.ndarray) -> float:
        """
        Measure true peak of audio signal.
        
        Note: This is a simplified sample peak measurement.
        True peak requires oversampling for accurate measurement.
        
        Args:
            y: Audio time series
            
        Returns:
            Peak amplitude in dBFS
        """
        peak_linear = np.max(np.abs(y))
        if peak_linear > 0:
            return 20 * np.log10(peak_linear)
        return -np.inf

    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """
        Normalize loudness and limit true peak.
        
        Process flow:
        1. Measure original integrated loudness (LUFS)
        2. Calculate and apply gain to reach target LUFS
        3. Check if peak exceeds max_true_peak
        4. If so, reduce gain to meet peak constraint
        5. Measure final loudness
        
        Args:
            y: Audio time series (numpy array, float32)
            sr: Sampling rate in Hz
            
        Returns:
            Tuple of (processed_audio, stats_dict)
            stats_dict contains:
                - original_lufs: Input loudness
                - final_lufs: Output loudness
                - original_peak_db: Input peak in dBFS
                - final_peak_db: Output peak in dBFS
                - gain_applied_db: Total gain applied
                - actions: List of actions taken
        """
        stats = {
            'original_lufs': None,
            'final_lufs': None,
            'original_peak_db': None,
            'final_peak_db': None,
            'gain_applied_db': 0.0,
            'actions': []
        }
        
        # Measure original loudness and peak
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        stats['original_lufs'] = loudness
        stats['original_peak_db'] = self.measure_true_peak(y)

        # Handle silence (loudness = -inf)
        if np.isinf(loudness):
            stats['actions'].append('skipped_silence')
            stats['final_lufs'] = loudness
            stats['final_peak_db'] = stats['original_peak_db']
            return y, stats

        # Calculate gain needed to reach target LUFS
        delta_lufs = self.target_lufs - loudness
        gain = 10.0 ** (delta_lufs / 20.0)
        
        # Apply gain
        y_normalized = y * gain
        
        # True Peak Limiting
        peak = np.max(np.abs(y_normalized))
        
        if peak > self._max_peak_linear:
            # Reduce gain to meet peak constraint
            limiter_gain = self._max_peak_linear / peak
            y_normalized = y_normalized * limiter_gain
            stats['actions'].append('peak_limited')
            # Update total gain
            gain = gain * limiter_gain
        
        # Record final stats
        stats['gain_applied_db'] = 20 * np.log10(gain)
        stats['final_lufs'] = meter.integrated_loudness(y_normalized)
        stats['final_peak_db'] = self.measure_true_peak(y_normalized)
        stats['actions'].append('normalized')
        
        return y_normalized, stats

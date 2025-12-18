"""
Audio Preprocessing Module for Music Information Retrieval.

This module provides tools for preprocessing audio files to ensure
consistency and fairness across tracks for MIR tasks like clustering.

Components:
    - AudioPreprocessor: Main orchestrator class
    - DurationHandler: Duration validation and cropping
    - LoudnessNormalizer: ITU-R BS.1770 loudness normalization

Standards:
    - ITU-R BS.1770-4: Loudness measurement
    - EBU R128: Loudness normalization

Example:
    >>> from src.audio_preprocessing import AudioPreprocessor
    >>> processor = AudioPreprocessor(target_lufs=-14.0)
    >>> stats = processor.process_directory("audio_files/")
"""

from .processor import AudioPreprocessor
from .duration_handler import DurationHandler
from .loudness_normalizer import LoudnessNormalizer

__all__ = ['AudioPreprocessor', 'DurationHandler', 'LoudnessNormalizer']

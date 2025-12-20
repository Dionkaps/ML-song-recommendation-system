"""
Audio Preprocessing Module for Music Information Retrieval.

This module provides tools for preprocessing audio files to ensure
consistency and fairness across tracks for MIR tasks like clustering.

Components:
    - AudioPreprocessor: Main orchestrator class
    - DurationHandler: Duration validation and cropping
    - LoudnessNormalizer: ITU-R BS.1770 loudness normalization
    - LoudnessScanner: Dataset-wide loudness analysis and target selection

Standards:
    - ITU-R BS.1770-4: Loudness measurement
    - EBU R128: Loudness normalization

Example:
    >>> from src.audio_preprocessing import AudioPreprocessor, LoudnessScanner
    >>> scanner = LoudnessScanner()
    >>> results = scanner.scan_directory("audio_files/")
    >>> analysis = scanner.analyze_distribution(results)
    >>> processor = AudioPreprocessor(target_lufs=analysis.suggested_target_lufs)
    >>> stats = processor.process_directory("audio_files/")
"""

from .processor import AudioPreprocessor
from .duration_handler import DurationHandler
from .loudness_normalizer import LoudnessNormalizer
from .loudness_scanner import LoudnessScanner, ScanResult, DistributionAnalysis

__all__ = [
    'AudioPreprocessor', 
    'DurationHandler', 
    'LoudnessNormalizer',
    'LoudnessScanner',
    'ScanResult',
    'DistributionAnalysis'
]


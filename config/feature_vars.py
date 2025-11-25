n_mfcc = 13
n_fft = 2048
hop_length = 512
n_mels = 128

include_genre = True
n_genres = 10 # Default fallback. Actual number is calculated dynamically from data.

# ---------------------------------------------------------------------
# Feature Weight Configuration (GLOBAL - applies to all clustering methods)
# ---------------------------------------------------------------------
# Control how much each feature group contributes to clustering.
# Higher values = MORE influence, lower values = LESS influence.
# 
# Feature groups (in order):
#   0. MFCC (Mel-frequency cepstral coefficients) - timbre/texture
#   1. Mel Spectrogram - frequency content over time
#   2. Spectral Centroid - brightness of sound
#   3. Zero Crossing Rate - noisiness/percussiveness
#   4. Genre (only used if include_genre=True) - actual genre labels
#
# Examples:
#   [1.0, 1.0, 1.0, 1.0]       - Equal contribution (when include_genre=False)
#   [2.0, 1.0, 0.5, 0.5]       - Prioritize MFCC, reduce spectral features
#   [3.0, 1.0, 1.0, 1.0, 0.0]  - Focus on timbre, ignore genre (when include_genre=True)
#
# NOTE: Only the first 4 values are used when include_genre=False (default)
#       All 5 values are used when include_genre=True

feature_group_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # [MFCC, MelSpec, SpectralCentroid, ZCR, Genre]
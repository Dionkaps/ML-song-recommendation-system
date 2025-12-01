n_mfcc = 13
n_fft = 2048
hop_length = 512
n_mels = 128

include_genre = True
n_genres = 10 # Default fallback. Actual number is calculated dynamically from data.

# ---------------------------------------------------------------------
# Feature Equalization Configuration (GLOBAL - applies to all clustering methods)
# ---------------------------------------------------------------------
# Method for equalizing feature contributions:
#   "pca_per_group" - Apply PCA to each feature group separately, reduce to same dims (RECOMMENDED)
#   "weighted"      - Use feature_group_weights to scale features (legacy method)
#
# With "pca_per_group", each feature group is:
#   1. Standardized (mean=0, std=1)
#   2. Reduced via PCA to `pca_components_per_group` dimensions
#   3. Concatenated together
# This ensures TRULY EQUAL contribution from each group regardless of original dimensionality.

feature_equalization_method = "pca_per_group"  # "pca_per_group" or "weighted"
pca_components_per_group = 5  # Number of PCA components per feature group (when using pca_per_group)

# ---------------------------------------------------------------------
# Feature Weight Configuration (only used when feature_equalization_method="weighted")
# ---------------------------------------------------------------------
# Control how much each feature group contributes to clustering.
# Higher values = MORE influence, lower values = LESS influence.
# 
# Feature groups (in order):
#   0. MFCC (Mel-frequency cepstral coefficients) - timbre/texture
#   1. Mel Spectrogram - frequency content over time
#   2. Spectral Centroid - brightness of sound
#   3. Spectral Flatness - tonal vs noisy characteristics
#   4. Zero Crossing Rate - noisiness/percussiveness
#   5. Genre (only used if include_genre=True) - actual genre labels
#
# Examples:
#   [1.0, 1.0, 1.0, 1.0, 1.0]       - Equal contribution (when include_genre=False)
#   [2.0, 1.0, 0.5, 0.5, 0.5]       - Prioritize MFCC, reduce spectral features
#   [3.0, 1.0, 1.0, 1.0, 1.0, 0.0]  - Focus on timbre, ignore genre (when include_genre=True)
#
# NOTE: Only the first 5 values are used when include_genre=False (default)
#       All 6 values are used when include_genre=True

feature_group_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # [MFCC, MelSpec, SpectralCentroid, SpectralFlatness, ZCR, Genre]
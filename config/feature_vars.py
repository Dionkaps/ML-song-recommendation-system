n_mfcc = 13
n_fft = 2048
hop_length = 512
n_mels = 128  # Kept for compatibility but not used
n_chroma = 12  # 12-dimensional pitch class profile

# ---------------------------------------------------------------------
# Supported baseline (2026-03-14)
# ---------------------------------------------------------------------
# The actively supported clustering baseline in this workspace is:
#   - audio-only clustering
#   - `spectral_plus_beat` handcrafted subset
#   - per-group StandardScaler + pca_per_group_5
#   - GMM as the default clustering method
#
# Genre remains metadata/evaluation-only, and MSD metadata is disabled by
# default until the unified songs.csv path is restored and validated.
supported_clustering_mode = "audio_only_spectral_plus_beat"
default_clustering_method = "gmm"

include_genre = False  # Keep genre only as metadata/evaluation, never as clustering input by default
n_genres = 10 # Default fallback. Actual number is calculated dynamically from data.

# Whether to include MSD (Million Song Dataset) metadata features
include_msd_features = False
msd_metadata_policy = "disabled_in_supported_baseline"
msd_metadata_restore_policy = (
    "restore_only_as_an_explicit_experiment_after_unified_metadata_audit_passes"
)

# Explicit handcrafted feature subset used for clustering.
clustering_feature_subset_name = "spectral_plus_beat"
clustering_audio_feature_keys = [
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flux",
    "spectral_flatness",
    "zero_crossing_rate",
    "beat_strength",
]

# Audio preprocessing invariants for the supported baseline.
baseline_target_duration_seconds = 29.0
baseline_target_lufs = -14.0
# Historical name retained for compatibility. The current preprocessing
# implementation applies a sample-peak ceiling rather than oversampled dBTP.
baseline_max_true_peak_dbtp = -1.0
baseline_sample_rate = 22050
baseline_output_subtype = "PCM_16"
baseline_force_mono = True

# ---------------------------------------------------------------------
# Explicit product decisions (2026-03-15)
# ---------------------------------------------------------------------
# Target cluster granularity:
# keep the product at broad macro-style clusters rather than micro-style
# fragmentation. The current supported GMM baseline should stay within 4..8
# occupied clusters, with 4 as the current reference target.
product_cluster_granularity_policy = "broad_macro_clusters"
product_cluster_target_min = 4
product_cluster_target_max = 8
product_cluster_target_default = 4

# Minimum acceptable stability gate for a production-default GMM model.
gmm_min_subsample_median_ari = 0.90
gmm_min_subsample_mean_ari = 0.75
gmm_min_reference_median_ari = 0.90
gmm_min_per_cluster_median_jaccard = 0.90

# Uncertain GMM assignments stay visible by default. Posterior-weighted ranking
# and hard thresholds remain available as optional operator controls, but the
# supported baseline keeps distance ranking as the product default because it
# preserves better proxy recommendation quality and catalog breadth.
uncertain_gmm_assignment_policy = "show_normally_with_optional_controls"
default_recommendation_ranking_method = "distance"
default_min_assignment_confidence = 0.0
default_min_selected_cluster_posterior = 0.0

# MSD numeric metadata should return only as an explicit experiment after
# coverage is nearly complete and the metadata path passes audit checks.
msd_restore_min_live_audio_coverage = 0.98
msd_restore_max_missing_audio_rows = 100
msd_restore_require_clean_schema_audit = True
msd_restore_require_explicit_experiment_profile = True
msd_restore_require_fresh_comparison_run = True
msd_restore_require_no_silent_fallback = True

# ---------------------------------------------------------------------
# Feature Groups for Extraction
# ---------------------------------------------------------------------
# AUDIO FEATURES (extracted from audio files via librosa):
#   1. MFCC (n_mfcc dimensions)
#   2. ΔMFCC - first derivatives (n_mfcc dimensions)
#   3. ΔΔMFCC - second derivatives (n_mfcc dimensions)
#   4. Spectral centroid (1 dimension)
#   5. Spectral rolloff (1 dimension)
#   6. Spectral flux (1 dimension)
#   7. Spectral flatness (1 dimension)
#   8. Zero Crossing Rate (1 dimension)
#   9. Chroma (12 dimensions)
#   10. Beat strength / onset rate (4 dimensions: tempo, mean/std onset strength, onset rate)
#
# MSD METADATA FEATURES (from Million Song Dataset HDF5 files):
#   11. Key (12-dim one-hot, 0=C to 11=B)
#   12. Mode (2-dim one-hot: major/minor)
#   13. Loudness (1-dim, normalized to 0-1)
#   14. Tempo from MSD (1-dim BPM)
#
# Total MSD dimensions: 16 (12 + 2 + 1 + 1)
# Note: Time signature, segments_timbre, and segments_pitches are excluded
#
# Each audio feature is summarized as mean + std, so dimensions are doubled for time-varying features.

# List of audio feature keys (for reference in clustering code)
AUDIO_FEATURE_KEYS = [
    "mfcc",           # MFCC
    "delta_mfcc",     # ΔMFCC (first derivatives)
    "delta2_mfcc",    # ΔΔMFCC (second derivatives)
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flux",
    "spectral_flatness",
    "zero_crossing_rate",
    "chroma",
    "beat_strength",
]

# MSD feature group names (4 groups, 16 dimensions total)
MSD_FEATURE_GROUPS = [
    "key",              # 12-dim one-hot
    "mode",             # 2-dim one-hot  
    "loudness",         # 1-dim (normalized)
    "msd_tempo",        # 1-dim
]

# Legacy alias
FEATURE_KEYS = AUDIO_FEATURE_KEYS

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
# AUDIO Feature groups (10 groups):
#   0. MFCC - timbre/texture
#   1. ΔMFCC - timbre dynamics (first derivative)
#   2. ΔΔMFCC - timbre acceleration (second derivative)
#   3. Spectral Centroid - brightness of sound
#   4. Spectral Rolloff - frequency distribution
#   5. Spectral Flux - rate of spectral change
#   6. Spectral Flatness - tonal vs noisy characteristics
#   7. Zero Crossing Rate - noisiness/percussiveness
#   8. Chroma - harmonic/pitch content
#   9. Beat Strength - rhythmic characteristics
#
# MSD Feature groups (4 groups, when include_msd_features=True):
#   10. Key - musical key (one-hot, 12-dim)
#   11. Mode - major/minor (one-hot, 2-dim)
#   12. Loudness (normalized, 1-dim)
#   13. MSD Tempo (1-dim)
#
# Genre (1 group, when include_genre=True):
#   14. Genre - actual genre labels (one-hot)

# Audio feature weights (10 groups)
audio_feature_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# MSD feature weights (4 groups: key, mode, loudness, tempo)
msd_feature_weights = [1.0, 1.0, 1.0, 1.0]

# Genre weight
genre_weight = 1.0

# Combined weights (for backward compatibility)
# This is: audio (10) + msd (4) + genre (1) = 15 groups maximum
feature_group_weights = audio_feature_weights + msd_feature_weights + [genre_weight]

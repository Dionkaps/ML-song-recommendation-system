# ML Song Recommendation System: Full Workspace Technical Report

Generated: 2026-03-14

## Scope

This report walks through the current workspace end to end:

- repository layout
- data sources and current dataset state
- audio preprocessing
- handcrafted feature extraction
- metadata / genre handling
- feature preparation and dimensionality
- clustering implementations
- saved experiment artifacts
- UI and visualization behavior
- important implementation gaps and mismatches

Excluded on purpose:

- pretrained embedding pipelines and models such as EnCodecMAE, MERT, MusiCNN, `musicnn_keras`, and related embedding extraction scripts

## 1. High-level repository layout

Main top-level areas:

- `src/`
  - `audio_preprocessing/`: active preprocessing pipeline
  - `features/`: handcrafted feature extraction plus MSD metadata feature support
  - `clustering/`: K-Means, GMM, HDBSCAN, VaDE
  - `ui/`: Tkinter + matplotlib + pygame UI
  - `data_collection/`: Million Song Dataset parsing and Deezer preview download
- `scripts/`
  - `analysis/`: clustering evaluation and validation helpers
  - `visualization/`: feature plotting
  - `utilities/`: cleanup / migration / maintenance helpers
  - wrapper runners for preprocessing, feature extraction, and all clustering
- `config/`
  - `feature_vars.py`: central handcrafted-feature and preprocessing config
- `data/`
  - current workspace has `millionsong_dataset.csv`
  - legacy backups live in `data/backup_old_csvs/`
- `audio_files/`
  - current local audio library
- `output/`
  - extracted handcrafted features
  - clustering results
  - clustering/feature-analysis metrics
  - experiment snapshots such as `optimal_configuration/`
- `docs/`
  - design docs, preprocessing docs, VaDE docs, generated reports
- `archived/`
  - deprecated scripts / old config assets

Main entrypoints:

- `run_pipeline.py`: intended all-in-one pipeline
- `scripts/run_audio_preprocessing.py`: preprocessing wrapper
- `scripts/run_feature_extraction.py`: handcrafted feature extraction wrapper
- `scripts/run_all_clustering.py`: runs K-Means, GMM, HDBSCAN, VaDE sequentially

## 2. What the project is actually doing

At its core, this workspace is a content-based music recommendation system built around:

1. collecting 30-second song previews
2. preprocessing them into a uniform audio format
3. extracting handcrafted MIR-style features
4. optionally augmenting those with Million Song Dataset metadata features
5. clustering songs into similarity groups
6. recommending songs from the same cluster in the UI

Important detail: the current UI recommends neighbors inside the same cluster using Euclidean distance in the 2D PCA projection, not distance in the full original feature space.

## 3. Current workspace state

### 3.1 Audio library state

Direct inspection of `audio_files/` shows:

- 5,535 audio files
- all files are `.wav`
- all files are mono (1 channel)
- all files are sampled at 22,050 Hz
- all files are `PCM_16`
- all files are exactly 29.0 seconds long
- no header-read errors were found in the current library

This means the current audio library is already fully normalized into a fixed MIR-friendly format.

### 3.2 Handcrafted feature artifact state

Direct inspection of `output/features/` shows:

- 55,350 `.npy` feature files
- 5,535 complete feature bundles
- 0 incomplete bundles
- 10 handcrafted feature files per song

Per-feature file counts:

| Feature | File count |
|---|---:|
| `mfcc` | 5535 |
| `delta_mfcc` | 5535 |
| `delta2_mfcc` | 5535 |
| `spectral_centroid` | 5535 |
| `spectral_rolloff` | 5535 |
| `spectral_flux` | 5535 |
| `spectral_flatness` | 5535 |
| `zero_crossing_rate` | 5535 |
| `chroma` | 5535 |
| `beat_strength` | 5535 |

Observed sample shapes for one current track (`!!! - Myth Takes`):

| Feature | Stored shape | Stored dtype |
|---|---|---|
| `mfcc` | `(13, 1249)` | `float32` |
| `delta_mfcc` | `(13, 1249)` | `float32` |
| `delta2_mfcc` | `(13, 1249)` | `float32` |
| `spectral_centroid` | `(1, 1249)` | `float64` |
| `spectral_rolloff` | `(1, 1249)` | `float64` |
| `spectral_flux` | `(1, 1249)` | `float32` |
| `spectral_flatness` | `(1, 1249)` | `float32` |
| `zero_crossing_rate` | `(1, 1249)` | `float64` |
| `chroma` | `(12, 1249)` | `float32` |
| `beat_strength` | `(4, 1)` | `float64` |

Because the current library is fixed at 29 s / 22,050 Hz / hop length 512, time-varying features naturally land around 1,249 frames.

### 3.3 Current clustering result artifacts

Current saved clustering result CSVs in `output/clustering_results/`:

- `audio_clustering_results_kmeans.csv`
- `audio_clustering_results_gmm.csv`
- `audio_clustering_results_hdbscan.csv`

There is no current `audio_clustering_results_vade.csv` in the workspace, so VaDE is implemented but not currently present as a saved output artifact.

Current row counts and cluster distributions:

| Result file | Rows | Cluster distribution |
|---|---:|---|
| K-Means | 5535 | `0: 1948`, `1: 3587` |
| GMM | 5535 | `0: 3041`, `1: 2494` |
| HDBSCAN | 5535 | `-1: 472`, `0: 6`, `1: 5051`, `2: 6` |

Interpretation:

- the current K-Means and GMM result files are both effectively 2-cluster solutions
- the current saved HDBSCAN result is highly imbalanced and mostly collapses into one giant cluster plus noise and two tiny micro-clusters

### 3.4 Current MSD / metadata state

Current `data/millionsong_dataset.csv`:

- 10,000 rows
- columns: `track_id`, `title`, `artist`, `genre`
- top split genre terms include `rock`, `hip hop`, `pop rock`, `blues-rock`, `pop`, `blues`, `jazz`, `ballad`

Current audio coverage relative to MSD CSV:

- MSD rows: 10,000
- local processed audio files: 5,535
- current local audio coverage is about 55.35% of the MSD CSV size

Current genre metadata cache:

- `output/features/genre_map.npy`: present
- `output/features/genre_list.npy`: present
- cached genre mapping entries: 5,574
- current audio files covered by cache: 5,535 / 5,535
- stale cached entries without matching audio: 39
- unique cached genres: 419

That means the workspace currently relies on cached genre metadata rather than an active unified `songs.csv`.

### 3.5 Legacy backup data assets

`data/backup_old_csvs/` contains the pre-unified metadata stack that the project used before moving toward `songs.csv`.

Current backup files:

- `songs_data_with_genre.csv`
  - columns: `title, artist, filename, genre`
  - legacy downloaded-song table
- `msd_features.csv`
  - columns include `track_id, artist, title, key, mode, loudness, tempo, key_confidence, mode_confidence, h5_path`
  - legacy numeric MSD feature table
- `msd_feature_vectors.npz`
  - cached numeric matrix generated from MSD metadata features
- `audio_msd_mapping.csv`
  - simple filename-to-track_id mapping table
- `msd_matches.csv`
  - detailed matched records between local audio and MSD entries
- `msd_unmatched.csv`
  - audio rows that could not be matched back to MSD

These files are important because they explain why the codebase has both:

- new unified-CSV logic
- and multiple legacy fallback paths still scattered through the repository

## 4. Data collection pipeline

### 4.1 Million Song Dataset parsing

Relevant files:

- `src/data_collection/extract_millionsong_dataset.py`
- `src/features/extract_msd_features.py`

There are two distinct MSD-related code paths:

1. `extract_millionsong_dataset.py`
   - extracts song metadata from the MSD tarball
   - walks `.h5` files
   - pulls title, artist, and artist terms as genre-like tags
   - writes `data/millionsong_dataset.csv`

2. `extract_msd_features.py`
   - extracts structured metadata features from MSD HDF5 analysis groups
   - supports building a unified `songs.csv`
   - turns MSD metadata into numeric vectors suitable for clustering

Important difference:

- `extract_millionsong_dataset.py` is about metadata/tag harvesting
- `extract_msd_features.py` is about numeric MSD features for clustering

### 4.2 Deezer preview downloading

Relevant file:

- `src/data_collection/deezer-song.py`

What it does:

- reads input tracks from `data/millionsong_dataset.csv`
- uses Deezer search API
- downloads preview URLs only
- writes audio files into `audio_files/`
- stores filename and source metadata
- uses a search cache (`deezer_search_cache.json`)
- uses a checkpoint file for resumability
- retries failed songs with exponential backoff

Important details:

- only songs whose `genre` field is present and not `unknown` are eligible for download
- downloaded filenames are sanitized for Windows
- previews are initially saved as `.mp3`
- the active preprocessing pipeline later converts them to `.wav`

Concurrency details:

- downloader uses `ThreadPoolExecutor`
- it caps worker count at 8
- each request path also adds a `0.5 s` sleep to reduce Deezer rate issues

## 5. Audio preprocessing pipeline

Relevant files:

- `src/audio_preprocessing/processor.py`
- `src/audio_preprocessing/duration_handler.py`
- `src/audio_preprocessing/loudness_normalizer.py`
- `scripts/run_audio_preprocessing.py`

### 5.1 Active preprocessing behavior

The active pipeline is the `AudioPreprocessor` class.

Default parameters:

- target duration: `29.0` seconds
- target loudness: `-14.0 LUFS`
- max peak target: `-1.0 dBTP`
- sample rate: `22050 Hz`

Per-file flow:

1. load audio with `librosa.load(..., sr=22050, mono=True)`
2. crop or reject based on duration
3. normalize loudness
4. write processed audio with `soundfile.write(..., subtype='PCM_16')`
5. if the original file was not WAV, replace it with a `.wav`

### 5.2 Duration handling

`DurationHandler` logic:

- if shorter than target duration: remove file
- if longer than target duration: keep the first 29 seconds
- if exactly target duration: leave unchanged

Very important detail:

- cropping is from the beginning, not centered and not from the end
- this assumes the intro is sufficiently representative

### 5.3 Loudness normalization

`LoudnessNormalizer` logic:

1. measure integrated loudness with `pyloudnorm` / BS.1770 meter
2. compute gain needed to reach target LUFS
3. apply gain
4. if resulting peak exceeds threshold, reduce gain again
5. save final loudness / peak stats

### 5.4 Technical caveat: "true peak" is simplified

The code and docs describe true-peak limiting and EBU R128 style control, but the implementation is technically simpler than a full true-peak limiter:

- peak measurement is `np.max(np.abs(y))`
- that is sample peak, not oversampled true peak
- limiting is a gain reduction step, not a dedicated brickwall true-peak limiter

So the intent is "true peak safe," but the actual implementation is closer to:

- integrated loudness normalization
- sample-peak constrained gain capping

### 5.5 Current processed-audio state

The current workspace confirms preprocessing has already happened successfully:

- all 5,535 audio files are mono
- all are 22,050 Hz
- all are PCM 16-bit WAV
- all are exactly 29.0 seconds

## 6. Handcrafted feature extraction

Relevant files:

- `src/features/extract_features.py`
- `config/feature_vars.py`
- `scripts/run_feature_extraction.py`
- `scripts/visualization/ploting.py`

### 6.1 Core extraction parameters

From `config/feature_vars.py`:

- `n_mfcc = 13`
- `n_fft = 2048`
- `hop_length = 512`
- `n_mels = 128`
- `n_chroma = 12`

Important note:

- `n_mels` is retained in config for compatibility, but the current handcrafted extractor does not actually use a mel-spectrogram feature in the clustering pipeline

### 6.2 Audio loading for feature extraction

`extract_features.py` uses:

- `soundfile.read()` for WAV files
- if multi-channel WAV is encountered, channels are averaged to mono
- `librosa.load(sr=None)` for non-WAV files

Meaning:

- feature extraction ultimately consumes mono audio
- in the current workspace, since all audio is already mono WAV at 22,050 Hz, extraction is fully consistent across tracks

### 6.3 Extracted handcrafted features

The extractor saves these 10 feature groups per track:

1. `mfcc`
2. `delta_mfcc`
3. `delta2_mfcc`
4. `spectral_centroid`
5. `spectral_rolloff`
6. `spectral_flux`
7. `spectral_flatness`
8. `zero_crossing_rate`
9. `chroma`
10. `beat_strength`

### 6.4 Exact algorithmic meaning of each feature

#### MFCC

- function: `librosa.feature.mfcc`
- parameters: `n_mfcc=13`
- role: timbre / spectral envelope summary
- raw stored shape per song: `(13, T)`
- summarized clustering contribution: `13 mean + 13 std = 26 dims`

#### Delta MFCC

- function: `librosa.feature.delta(mfcc)`
- role: first-order timbral dynamics
- raw shape: `(13, T)`
- summarized contribution: `26 dims`

#### Delta2 MFCC

- function: `librosa.feature.delta(mfcc, order=2)`
- role: second-order timbral acceleration
- raw shape: `(13, T)`
- summarized contribution: `26 dims`

#### Spectral centroid

- function: `librosa.feature.spectral_centroid`
- role: brightness / center of mass of spectrum
- raw shape: `(1, T)`
- summarized contribution: `2 dims`

#### Spectral rolloff

- function: `librosa.feature.spectral_rolloff`
- parameter: `roll_percent=0.85`
- role: frequency below which 85% of spectral energy is contained
- raw shape: `(1, T)`
- summarized contribution: `2 dims`

#### Spectral flux

- implemented via `librosa.onset.onset_strength`
- role: amount of spectral change over time
- raw shape: `(1, T)`
- summarized contribution: `2 dims`

#### Spectral flatness

- function: `librosa.feature.spectral_flatness`
- role: tonal vs noisy character
- raw shape: `(1, T)`
- summarized contribution: `2 dims`

#### Zero-crossing rate

- function: `librosa.feature.zero_crossing_rate`
- role: noisiness / percussiveness proxy
- raw shape: `(1, T)`
- summarized contribution: `2 dims`

#### Chroma

- function: `librosa.feature.chroma_stft`
- parameters: `n_chroma=12`
- role: 12 pitch-class energy bands
- raw shape: `(12, T)`
- summarized contribution: `12 mean + 12 std = 24 dims`

#### Beat strength

Constructed rhythm block, not a standard single librosa array:

- onset envelope from `librosa.onset.onset_strength`
- tempo from `librosa.feature.tempo`
- onset count from `librosa.onset.onset_detect`

Stored as 4 scalar values:

1. tempo
2. mean onset strength
3. std onset strength
4. onset rate (onsets per second)

Raw stored shape:

- `(4, 1)`

Clustering contribution:

- `4 dims`

### 6.5 Exact handcrafted feature dimensionality

After per-track time summarization:

| Feature group | Dims |
|---|---:|
| MFCC | 26 |
| Delta MFCC | 26 |
| Delta2 MFCC | 26 |
| Spectral centroid | 2 |
| Spectral rolloff | 2 |
| Spectral flux | 2 |
| Spectral flatness | 2 |
| Zero-crossing rate | 2 |
| Chroma | 24 |
| Beat strength | 4 |
| Total handcrafted audio dims | 116 |

That `116`-dimensional audio-only vector is the base handcrafted representation used throughout the workspace.

### 6.6 Feature storage design

Features are saved per song, per feature group:

- filename pattern: `<song_base>_<feature_key>.npy`
- writes are atomic through a temporary file then `os.replace`
- extractor supports resume / skip-complete-bundle behavior

### 6.7 Parallel extraction design

`run_feature_extraction()`:

- supports `.wav`, `.mp3`, `.flac`, `.m4a`
- chooses thread executor on Windows and process executor elsewhere by default
- caps thread workers at 8
- records failure logs to `feature_extraction_failures.txt` if needed

## 7. Genre and metadata handling

Relevant files:

- `src/utils/genre_mapper.py`
- `src/features/extract_msd_features.py`
- `config/feature_vars.py`

### 7.1 Genre handling

The codebase supports multi-label genre strings from CSV, but clustering uses genre as metadata by default, not as an input feature.

Config defaults:

- `include_genre = False`
- this is intentionally set to avoid genre leakage into clustering

How genre mapping works:

- CSV genre strings are split on commas
- a multi-label mapping can be built with `MultiLabelBinarizer`
- for clustering outputs and evaluation, only the primary genre is used
- primary genre = first genre in the list

Important consequence:

- even when source metadata is multi-label, clustering metrics and output CSVs are effectively single-label at the genre-reference level

### 7.2 MSD metadata features

The codebase supports 4 structured MSD metadata groups:

1. key: 12-dim one-hot
2. mode: 2-dim one-hot
3. loudness: 1 dim
4. tempo: 1 dim

Total MSD feature dims:

- `16`

From `extract_msd_features.py`:

- key is read from MSD analysis and one-hot encoded
- mode is encoded as minor/major one-hot
- loudness is numeric
- tempo is numeric

In the standalone MSD feature builder, loudness and tempo are min-max normalized.

In clustering dataset assembly (`kmeans.py` path), loudness and tempo are z-scored using the currently loaded song set.

### 7.3 Current workspace metadata reality

Config says:

- `include_msd_features = True`

But current workspace actually lacks:

- `data/songs.csv`
- `data/songs_data_with_genre.csv`

So in the current checked-in workspace:

- cached genre metadata exists and covers the current audio files
- unified numeric MSD metadata is not currently available as an active source file
- many saved experiment artifacts explicitly report `include_msd_features: false`

This means:

- the codebase supports MSD metadata features
- the current workspace state is primarily audio-only for practical clustering runs

## 8. Feature preparation before clustering

Relevant file:

- `src/clustering/kmeans.py`

This file contains the shared feature assembly logic used by all clustering algorithms.

### 8.1 How per-track vectors are built

For each song:

1. load the 10 handcrafted `.npy` arrays
2. summarize time-varying arrays as `mean` and `std`
3. flatten the 4-value `beat_strength` block directly
4. optionally append 16 MSD metadata dims
5. optionally append one-hot genre dims

Base dimensionalities:

- audio-only: `116 dims`
- audio + MSD: `132 dims`
- audio + MSD + genre: `132 + n_genres`

With the current cached genre set of 419 unique genres, a full audio+MSD+genre vector would be:

- `132 + 419 = 551 dims`

However, the default config keeps genre features off.

### 8.2 Equalization strategy

Global config:

- `feature_equalization_method = "pca_per_group"`
- `pca_components_per_group = 5`

Two supported methods exist:

1. `pca_per_group`
2. `weighted`

#### Method A: `pca_per_group`

For each feature group independently:

1. standardize group with `StandardScaler`
2. if group dims > target dims, reduce with PCA
3. if group dims < target dims, zero-pad
4. normalize group to equal average Frobenius norm
5. concatenate all transformed groups

This is a strong design choice: it forces equal contribution per feature group, rather than letting big groups like MFCC dominate by dimensionality alone.

Current dimensionality under `pca_per_group = 5`:

- audio-only: `10 groups x 5 = 50 dims`
- audio + MSD: `14 groups x 5 = 70 dims`
- audio + MSD + genre: `15 groups x 5 = 75 dims`

#### Method B: `weighted`

Legacy alternative:

1. z-score all raw concatenated features
2. build per-dimension weights from group weights
3. scale each group by `multiplier / sqrt(group_size)`

Default raw group weights are all `1.0`, so the weighting path mostly exists as a compatibility option.

## 9. Clustering algorithms

All clustering methods share:

- the same `load_clustering_dataset()` assembly path
- the same PCA-based 2D visualization basis via `compute_visualization_coords()`
- genre kept as metadata for reporting even when not used as input

### 9.1 K-Means

Relevant file:

- `src/clustering/kmeans.py`

Implementation details:

- algorithm: `sklearn.cluster.KMeans`
- selection metric: silhouette score
- selection model init: `n_init=20`
- final fit init: `n_init=10`
- `random_state=42`

Dynamic cluster selection:

- code computes a heuristic range using `compute_cluster_range()`
- base heuristic: `sqrt(n_samples / 2)`
- capped at 30 or 50 depending on available hints

For the current 5,535-song workspace, the code heuristic would permit a broad search range, but the saved `output/metrics/kmeans_selection_criteria.csv` artifact only evaluates `K = 2..12`, so that saved artifact comes from a narrower experimental run than the broadest code path.

Current saved K-Means diagnostics:

- best saved silhouette in `kmeans_selection_criteria.csv`: `K=2`, silhouette `0.204884`

Current saved K-Means result file:

- 2 clusters
- sizes `1948` and `3587`

What K-Means outputs per song:

- cluster label
- distance to assigned centroid in prepared feature space
- shared PCA coordinates `PCA1`, `PCA2`

### 9.2 Gaussian Mixture Model (GMM)

Relevant file:

- `src/clustering/gmm.py`

Implementation details:

- algorithm: `sklearn.mixture.GaussianMixture`
- default covariance type: `full`
- `max_iter=200`
- `tol=1e-3`
- `init_params="kmeans"`
- `reg_covar=1e-5`
- `random_state=42`

Component selection:

- primary metric: BIC
- tie-break behavior: if several models are within `10.0` BIC points of the best BIC, prefer the one with the best silhouette

Current saved GMM diagnostics:

- components evaluated in saved artifact: `2..12`
- lowest saved BIC occurs at `2 components`
- saved silhouette at 2 components: `0.142858`

Current saved GMM result file:

- 2 components
- cluster sizes `3041` and `2494`

Per-song outputs:

- hard cluster assignment via `predict`
- max posterior probability as `Confidence`
- sample log-likelihood as `LogLikelihood`
- PCA coordinates

### 9.3 HDBSCAN

Relevant file:

- `src/clustering/hdbscan.py`

Implementation details:

- algorithm: `hdbscan.HDBSCAN`
- supports noise label `-1`
- selection metric is a custom score:
  - DBCV
  - minus a noise penalty (`0.20 * noise_fraction`)
  - plus a small silhouette term (`0.05 * silhouette_non_noise`)

Dynamic search space:

- candidate `min_cluster_size` values are built from fixed values plus dataset-size fractions
- per `min_cluster_size`, candidate `min_samples` values include conservative and permissive options

Current saved HDBSCAN diagnostics:

- best saved score in `hdbscan_selection_criteria.csv` is at:
  - `min_cluster_size = 5`
  - `min_samples = 1`
  - `3 clusters`
  - `noise_fraction = 0.0853`
  - `dbcv = 0.03229`
  - `silhouette = 0.20019`

Current saved HDBSCAN result file:

- noise points: `472` (`8.53%`)
- cluster sizes: `6`, `5051`, `6`

This saved run is technically valid but practically weak because it is dominated by one giant cluster.

Per-song outputs:

- cluster label (including `-1` for noise)
- HDBSCAN membership probability
- distance to simple mean centroid of assigned cluster
- PCA coordinates

### 9.4 VaDE

Relevant file:

- `src/clustering/vade.py`

VaDE is implemented, but a current saved result CSV is not present in this workspace.

Architecture:

- encoder hidden layers: `[512, 256]` by default
- decoder hidden layers: `[256, 512]` by default
- latent dimension: `10`
- cluster prior: learnable diagonal-covariance GMM in latent space

Key training flow:

1. pretrain autoencoder
2. encode data into latent means
3. run BIC-based GMM selection in latent space
4. initialize VaDE cluster prior from sklearn GMM
5. train full VaDE with KL warmup

Default run-time hyperparameters in `run_vade_clustering()`:

- `latent_dim = 10`
- `batch_size = 128`
- `lr = 1e-3`
- `pretrain_epochs = 20`
- `train_epochs = 80`
- `kl_warmup_epochs = 20`
- `kl_c_weight = 0.1`

Important details:

- cluster assignments are based on posterior responsibilities over the latent mean, not sampled latent noise
- VaDE uses the same 2D PCA projection of prepared features for output visualization consistency
- selection in VaDE is done in latent space, which is the correct place for its GMM prior

## 10. Current experiment artifacts and what they say

### 10.1 Root metrics folder

The root metrics files show that the workspace has been actively used for feature sensitivity and equalization experiments, not just one default clustering pass.

Important top-level experiment files:

- `output/metrics/feature_combo_clustering_summary.csv`
- `output/metrics/pca_sweep_summary_audio_only.csv`
- `output/metrics/feature_group_variance_summary.csv`
- `output/metrics/feature_group_correlation_summary.csv`

### 10.2 Feature group variance summary

From `feature_group_variance_summary.csv`:

- spectral rolloff and spectral centroid have by far the largest raw variance magnitudes
- MFCC has moderate but wide-ranging variance
- chroma and spectral flatness are low-variance groups

This strongly justifies the use of per-group scaling / PCA equalization. Without equalization, spectral magnitude-style groups can dominate.

### 10.3 Feature group correlation summary

From `feature_group_correlation_summary.csv`, strongest mean absolute group correlations include:

- spectral flux <-> beat strength
- spectral centroid <-> zero-crossing rate
- spectral centroid <-> spectral rolloff
- spectral flatness <-> zero-crossing rate

Interpretation:

- several spectral groups are redundant or partially overlapping
- beat/rhythm information is strongly linked with onset-based spectral change
- this again supports groupwise dimensionality control instead of naive concatenation

### 10.4 Audio-only PCA-per-group sweep

From `pca_sweep_summary_audio_only.csv`:

| PCA comps per group | Total dims | K-Means best silhouette | GMM min-BIC components | GMM silhouette at min BIC | HDBSCAN best score |
|---:|---:|---:|---:|---:|---:|
| 2 | 20 | 0.24751 | 7 | 0.08434 | -0.08999 |
| 5 | 50 | 0.22393 | 4 | 0.09016 | -0.00912 |
| 8 | 80 | 0.21250 | 2 | 0.15550 | 0.02139 |
| 12 | 120 | 0.20488 | 2 | 0.14286 | 0.02525 |

Interpretation:

- fewer PCA components per group help K-Means silhouette
- more retained dimensions help GMM and HDBSCAN somewhat
- there is no single universally best setting across all algorithms

### 10.5 Feature sensitivity suite

`output/metrics/feature_sensitivity_suite/run_metadata.json` reports:

- total songs: `5535`
- raw audio dims: `116`
- feature combos tested: `20`
- preprocess modes tested: `3`
- total scenarios: `60`

Best full-evaluation candidates from `top_candidates_full_eval.csv`:

| Combo | Preprocess mode | Transformed dims | K-Means silhouette | GMM silhouette | HDBSCAN score |
|---|---|---:|---:|---:|---:|
| `timbre_plus_spectral` | `pca_per_group_2` | 16 | 0.27029 | 0.05705 | 0.11090 |
| `spectral_plus_beat` | `pca_per_group_5` | 30 | 0.26433 | 0.15276 | 0.20002 |
| `all_audio` | `pca_per_group_5` | 50 | 0.22393 | 0.09040 | -0.00912 |
| `spectral_plus_beat` | `raw_zscore` | 14 | 0.27383 | 0.07388 | -0.03470 |

Interpretation:

- all-audio is not automatically best
- spectral + rhythm features are especially effective in this workspace
- `spectral_plus_beat` with `pca_per_group_5` looks like the strongest cross-method compromise

### 10.6 Optimal configuration snapshot

`output/optimal_configuration/run_20260311_212955/run_summary.json` records a selected configuration:

- combo: `spectral_plus_beat`
- groups:
  - `spectral_centroid`
  - `spectral_rolloff`
  - `spectral_flux`
  - `spectral_flatness`
  - `zero_crossing_rate`
  - `beat_strength`
- preprocess mode: `pca_per_group_5`
- samples: `5535`
- dims: `30`

Recorded results for that run:

- K-Means: `2 clusters`, silhouette `0.26433`
- GMM: `4 components`, silhouette `0.15276`, avg confidence `0.93790`
- HDBSCAN: `2 clusters`, noise fraction `0.04661`, silhouette on non-noise `0.19943`

This is the clearest sign in the workspace that the project has moved away from "just throw all handcrafted features in" toward a more selective feature set.

## 11. Evaluation and validation scripts

Relevant files:

- `scripts/analysis/evaluate_clustering.py`
- `scripts/analysis/compare_bic_aic.py`
- `scripts/analysis/validate_gmm.py`

### 11.1 Evaluation metrics used

The evaluation script supports:

- Silhouette score
- Calinski-Harabasz
- Davies-Bouldin
- Adjusted Rand Index
- Normalized Mutual Information
- stability ARI across reruns / bootstrap runs

Important evaluation behavior:

- HDBSCAN noise points are excluded from internal metrics
- true labels are derived from primary genre only
- stability is computed differently per algorithm:
  - K-Means: reruns across seeds
  - GMM: reruns across seeds
  - HDBSCAN: bootstrap subsampling
  - VaDE: repeated training runs with reduced epochs

### 11.2 Stability snapshot

From `output/metrics/many_tests_2026-03-11_20-50-02/stability_summary.csv`:

| Config | K-Means stability ARI | GMM stability ARI |
|---|---:|---:|
| `full_audio_pca5` | 0.99767 | 0.97641 |
| `spectral_only` | 0.98979 | 0.96501 |
| `timbre_only` | 1.00000 | 1.00000 |

Interpretation:

- these clustering solutions are very reproducible
- stability is not the main bottleneck; discriminative usefulness is

## 12. Visualization and UI behavior

Relevant files:

- `scripts/visualization/ploting.py`
- `src/ui/modern_ui.py`

### 12.1 Feature plotting

`ploting.py`:

- loads raw audio
- loads stored feature arrays
- renders per-song plots into `output/plots/`

Supported plot types:

- MFCC
- Delta MFCC
- Delta2 MFCC
- spectral centroid
- spectral rolloff
- spectral flux
- spectral flatness
- zero-crossing rate
- chroma
- beat_strength
- legacy `melspectrogram` if a file exists

Important note:

- the plotting script still includes a mel-spectrogram code path even though the active handcrafted extractor does not produce mel-spectrogram `.npy` files by default

### 12.2 UI recommendation mechanics

The UI:

- is Tkinter based
- embeds a matplotlib scatter plot
- uses pygame for audio playback
- lists all songs
- supports search
- shows cluster plot and recommendations

Recommendation logic:

1. pick selected song
2. find other songs with the same cluster label
3. compute nearest neighbors in the 2D PCA coordinates
4. recommend top `N` nearest cluster-mates

That means recommendations are:

- cluster-constrained
- 2D-PCA-distance-based

They are not:

- nearest neighbors in the full 50 / 116 / 132 dimensional feature space
- learned ranking predictions

### 12.3 Standalone inspection and maintenance scripts

There are several non-pipeline helper scripts in the repo that are easy to miss but are still part of the workspace story:

- `check_audio_lengths.py`
  - scans audio durations and reports deviations from the 29-second target
- `check_feature_dims.py`
  - loads sample feature files and prints stored array shapes
- `extract_sample_20.py`
  - extracts handcrafted features for a 20-file sample for quick testing
- `explore_msd_structure.py`
  - inspects the HDF5 structure of the Million Song Dataset and prints available fields

Maintenance / hygiene utilities in `scripts/utilities/`:

- `cleanup_orphaned_features.py`
  - removes feature bundles whose matching audio files no longer exist
- `cleanup_orphaned_files.py`
  - removes old orphaned `.mp3` files that do not appear in CSV metadata
- `check_problem_songs.py`
  - spot-checks songs that produce warnings and confirms whether audio, CSV rows, and features exist
- `add_track_ids_to_csv.py`
  - backfills `track_id` into `millionsong_dataset.csv` by matching against HDF5 files
- `create_preprocessing_report.py`
  - generates before/after preprocessing CSV reports
- `migrate_to_unified_csv.py`
  - merges old CSVs into a single `songs.csv`

These scripts are not the main recommendation path, but they are a major part of how the workspace has been debugged, cleaned, and migrated over time.

### 12.4 Present but intentionally excluded files

Because you explicitly asked to exclude pretrained-model coverage, I am not treating these as part of the report body even though they are present in the workspace:

- `run_extraction.py`
- `src/features/extract_embeddings.py`
- `scripts/extract_encodecmae.py`
- `scripts/extract_mert.py`
- `scripts/extract_musicnn.py`
- `scripts/utilities/check_outputs.py`
- `scripts/utilities/verify_models.py`
- embedding-related docs such as `docs/EMBEDDING_EXTRACTION.md`, `docs/AUDIO_EMBEDDING_EXTRACTION.md`, and `docs/QUICKSTART_EMBEDDINGS.md`

I am calling them out here only so the report is explicit about what is present versus what was intentionally left out.

## 13. Important current workspace mismatches and technical caveats

This section matters because the repository contains both active logic and stale / partially migrated logic.

### 13.1 `songs.csv` is expected by clustering, but absent in this workspace

Current state:

- `config/feature_vars.py` sets `include_msd_features = True`
- clustering wrappers pass `songs_csv_path="data/songs.csv"`
- `data/songs.csv` is not present

Consequence:

- MSD metadata is implemented in the codebase
- but it is not currently available as an active data source in this workspace
- many experiment artifacts explicitly disabled MSD features

### 13.2 Genre metadata currently comes from cache, not a live unified CSV

Because neither `data/songs.csv` nor `data/songs_data_with_genre.csv` exists now, the active genre information in current outputs is effectively coming from cached files:

- `output/features/genre_map.npy`
- `output/features/genre_list.npy`

That cache fully covers the current audio library but also contains 39 stale entries.

### 13.3 `loudness_scanner` is referenced but missing

`run_pipeline.py` and `src/audio_preprocessing/__init__.py` reference:

- `src.audio_preprocessing.loudness_scanner`
- `LoudnessScanner`
- `save_scan_results`

But there is no `loudness_scanner.py` present in `src/audio_preprocessing/`.

Consequence:

- the intended adaptive loudness-scan stage described in `run_pipeline.py` is not fully present in the workspace
- the simpler active preprocessing path via `AudioPreprocessor` still exists and is usable

### 13.4 MSD extraction scripts and current CSV schema are out of sync

Current `data/millionsong_dataset.csv` has columns:

- `track_id`, `title`, `artist`, `genre`

But `extract_millionsong_dataset.py` writes only:

- `title`, `artist`, `genre`

This suggests the current data artifact was produced by a different or evolved script path than the checked-in extractor code.

Related issue:

- `scripts/utilities/migrate_to_unified_csv.py` assumes a 3-column MSD input and renames columns to `['msd_title', 'msd_artist', 'genre']`
- that assumption does not match the current 4-column CSV

### 13.5 There are legacy / stale references elsewhere too

Examples:

- the README still mentions WKBSC, but no active `scripts/wkbsc.py` exists in the current workspace
- `ploting.py` still carries legacy mel-spectrogram handling
- `src/utils/audio_normalizer.py` is an older simpler duration-only normalizer beside the newer preprocessing package
- some utility scripts still assume `.mp3` audio files, while the current processed library is entirely `.wav`

## 14. Practical summary

If we describe the project as it exists right now, without the excluded pretrained model stack, the most accurate summary is:

- the audio side of the workspace is in a very clean state: 5,535 mono 22.05 kHz 29-second WAV files
- handcrafted MIR features are fully extracted for all current tracks
- the handcrafted audio-only representation is 116 dims before equalization
- the project has a sophisticated per-group PCA equalization path that reduces feature dominance issues
- the clustering codebase includes four methods: K-Means, GMM, HDBSCAN, and VaDE
- the current saved result artifacts only include K-Means, GMM, and HDBSCAN
- the workspace's strongest experiment evidence points toward a reduced `spectral_plus_beat` feature set rather than the full 116-dim handcrafted vector
- metadata / genre support exists, but the unified `songs.csv` path is currently missing, so the repository is in a partially migrated state

## 15. File map for the most important active pieces

Core pipeline:

- `run_pipeline.py`
- `scripts/run_audio_preprocessing.py`
- `scripts/run_feature_extraction.py`
- `scripts/run_all_clustering.py`

Preprocessing:

- `src/audio_preprocessing/processor.py`
- `src/audio_preprocessing/duration_handler.py`
- `src/audio_preprocessing/loudness_normalizer.py`

Features:

- `src/features/extract_features.py`
- `src/features/extract_msd_features.py`
- `config/feature_vars.py`

Clustering:

- `src/clustering/kmeans.py`
- `src/clustering/gmm.py`
- `src/clustering/hdbscan.py`
- `src/clustering/vade.py`

Metadata / genres:

- `src/utils/genre_mapper.py`
- `data/millionsong_dataset.csv`
- `output/features/genre_map.npy`
- `output/features/genre_list.npy`

Evaluation / experiments:

- `scripts/analysis/evaluate_clustering.py`
- `output/metrics/feature_sensitivity_suite/run_metadata.json`
- `output/metrics/feature_sensitivity_suite/top_candidates_full_eval.csv`
- `output/optimal_configuration/run_20260311_212955/run_summary.json`

UI / plotting:

- `src/ui/modern_ui.py`
- `scripts/visualization/ploting.py`

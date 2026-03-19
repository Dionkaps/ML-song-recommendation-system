# ML Song Recommendation System

A machine learning project that analyzes audio files to create a content-based music recommendation system using deep learning embeddings and clustering algorithms.

## Current Supported Baseline

The actively supported clustering baseline in this workspace is:

- audio-only clustering
- handcrafted `spectral_plus_beat` features for clustering
- per-group `StandardScaler` + `pca_per_group_5`
- `GMM` as the default clustering method
- genre kept as metadata only
- MSD metadata disabled by default until `data/songs.csv` is restored
- preprocessing invariants: mono, `22050 Hz`, `29s`, `PCM_16`, loudness-normalized

Historical sections below still mention older and experimental paths. Use the supported baseline above as the source of truth for the current workspace state.

See [docs/SUPPORTED_BASELINE.md](docs/SUPPORTED_BASELINE.md) for the detailed baseline contract.
See [docs/RECOMMENDED_PRODUCTION_BASELINE.md](docs/RECOMMENDED_PRODUCTION_BASELINE.md) for the short production-default summary and explicit comparison baselines.
See [docs/DECISION_POLICY.md](docs/DECISION_POLICY.md) for the explicit cluster-granularity, stability-gate, uncertainty-handling, and MSD-return decisions.

## Key Features

- **🎵 Deep Audio Embeddings**: Extract rich audio representations using state-of-the-art models (EnCodecMAE, MERT, MusiCNN)
- **🔬 Multiple Clustering Algorithms**: K-Means, GMM, HDBSCAN, and VaDE (deep learning)
- **🎨 Interactive UI**: Browse songs, get recommendations, play audio
- **📊 Visualization**: PCA-based 2D visualization of clusters

## Project Structure

```
ML-song-recommendation-system/
│
├── src/                          # Main source code
│   ├── clustering/               # Clustering algorithms (K-Means, GMM, VaDE, HDBSCAN)
│   ├── features/                 # Feature extraction modules
│   ├── ui/                       # User interface modules
│   └── data_collection/          # Data collection scripts
│
├── scripts/                      # Scripts
│   ├── analysis/                 # Analysis scripts
│   └── utilities/                # Maintenance and utility scripts
│
├── tests/                        # Test suite
│
├── docs/                         # Documentation and reports
│   ├── reports/                  # Generated reports
│   └── ...                       # Implementation guides and references
│
├── config/                       # Configuration files
│
├── archived/                     # Deprecated or unused files
│
├── output/                       # Output files and results
│
├── run_pipeline.py               # Main pipeline entry point
├── run_extraction.py             # Feature extraction entry point
└── requirements.txt              # Project dependencies

```

## Audio Embedding Extraction (NEW)

This project now supports extracting deep audio embeddings using three pretrained models:

| Model | Description | Embedding Dim | Python |
|-------|-------------|---------------|--------|
| **EnCodecMAE** | Self-supervised audio encoder | 768 | 3.12 |
| **MERT** | Music understanding transformer | 768 | 3.12 |
| **MusiCNN** | Music tagging CNN | 753 | 3.7 |

### Quick Setup

```powershell
# See available extraction commands
python run_extraction.py

# Check extraction status
python run_extraction.py --status
```

📖 **Full Guide**: [docs/AUDIO_EMBEDDING_EXTRACTION.md](docs/AUDIO_EMBEDDING_EXTRACTION.md)  
⚡ **Quick Start**: [docs/QUICKSTART_EMBEDDINGS.md](docs/QUICKSTART_EMBEDDINGS.md)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install numpy pandas scikit-learn librosa matplotlib pygame hdbscan torch
```

### 2. Run the Complete Pipeline
```bash
python run_pipeline.py
```

This will:
- Optionally download and preprocess audio in `audio_files/`
- Extract handcrafted audio features
- Generate feature visualizations
- Run the default GMM clustering pipeline
- Produce clustering outputs used by the recommendation UI

### 3. Choose Clustering Method
```bash
# Gaussian Mixture Model (default supported baseline)
python run_pipeline.py --clustering-method gmm

# K-Means
python run_pipeline.py --clustering-method kmeans

# HDBSCAN (density-based)
python run_pipeline.py --clustering-method hdbscan

# VaDE (deep learning)
python run_pipeline.py --clustering-method vade
```

### 4. Skip Specific Steps
```bash
python run_pipeline.py --skip extract plot
```

## Features

- **Multiple Clustering Algorithms**: K-Means, GMM, HDBSCAN, and VaDE (deep learning)
- **Rich Audio Features**: MFCC families, spectral descriptors, chroma, and beat/rhythm summaries
- **Interactive UI**: Browse songs, get recommendations, play audio
- **Visualization**: PCA-based 2D visualization of clusters
- **Deep Learning**: VaDE combines VAE and GMM for joint feature learning and clustering

## Module Details

### Clustering Algorithms
- **K-Means** (`src/clustering/kmeans.py`): Fast centroid-based clustering with hard assignments
- **GMM** (`src/clustering/gmm.py`): Probabilistic clustering with Gaussian Mixture Models
- **HDBSCAN** (`src/clustering/hdbscan.py`): Density-based clustering with noise detection
- **VaDE** (`src/clustering/vade.py`): Deep learning approach combining VAE and GMM

### Feature Extraction
The system extracts these audio features:

**Traditional Features:**
- **MFCC** (Mel-frequency cepstral coefficients)
- **Delta / Delta-Delta MFCC**
- **Spectral centroid**
- **Spectral rolloff**
- **Spectral flux**
- **Spectral flatness**
- **Zero-crossing rate**
- **Chroma**
- **Beat strength / onset summaries**

**Deep Learning Embeddings:**
- **EnCodecMAE** - Self-supervised audio representations (768-dim)
- **MERT** - Music-specific transformer features (768-dim)
- **MusiCNN** - Music tagging features (753-dim)

See [Audio Embedding Extraction Guide](docs/AUDIO_EMBEDDING_EXTRACTION.md) for details.

### Configuration
Edit `config/feature_vars.py` to adjust:
```python
n_mfcc = 13      # Number of MFCC coefficients
n_fft = 2048     # FFT window size
hop_length = 512 # Hop length for STFT
n_mels = 128     # Compatibility-only; the active handcrafted baseline does not use mel-spectrogram features
```

## Usage Examples

### Extract Features Only
```bash
python src/features/extract_features.py
```

### Run Clustering Algorithms Directly
```bash
# K-Means
python src/clustering/kmeans.py

# Gaussian Mixture Model
python src/clustering/gmm.py

# HDBSCAN
python src/clustering/hdbscan.py

# VaDE (requires PyTorch)
python src/clustering/vade.py
```

### Visualize Features
```bash
python scripts/visualization/ploting.py --features_dir output/features --plots_dir output/plots
```

## Output Files

- `output/embeddings/`: Deep learning embeddings (.npy files)
  - `encodecmae/` - EnCodecMAE embeddings
  - `mert/` - MERT embeddings
  - `musicnn/` - MusiCNN embeddings
- `output/features/`: Traditional extracted features (.npy files)
- `output/clustering_results/`: Clustering results (.csv files)
  - `audio_clustering_results_kmeans.csv`
  - `audio_clustering_results_gmm.csv`
  - `audio_clustering_results_hdbscan.csv`
  - `audio_clustering_results_vade.csv`

## Requirements

- Python 3.12 (for EnCodecMAE and MERT)
- Python 3.7 (for MusiCNN - due to numpy<1.17 requirement)
- NumPy
- Pandas
- scikit-learn
- librosa
- matplotlib
- pygame
- hdbscan
- PyTorch (for VaDE and modern embeddings)
- TensorFlow 2.7.4 (for MusiCNN)
- transformers (for MERT)
- tkinter (usually comes with Python)

See `requirements.txt` for specific versions.

## Documentation

| Document | Description |
|----------|-------------|
| [Audio Embedding Extraction](docs/AUDIO_EMBEDDING_EXTRACTION.md) | Complete guide to extracting deep audio embeddings |
| [Quick Start: Embeddings](docs/QUICKSTART_EMBEDDINGS.md) | 5-minute setup for embedding extraction |
| [VaDE Implementation](docs/VADE_IMPLEMENTATION.md) | Deep learning clustering with VAE+GMM |
| [Supported Baseline](docs/SUPPORTED_BASELINE.md) | Current supported clustering and preprocessing contract |
| [Recommended Production Baseline](docs/RECOMMENDED_PRODUCTION_BASELINE.md) | Short production-default summary plus explicit comparison baselines |
| [Decision Policy](docs/DECISION_POLICY.md) | Explicit product decisions for cluster granularity, stability gates, uncertainty handling, and MSD restoration |

## License

See LICENSE file for details.

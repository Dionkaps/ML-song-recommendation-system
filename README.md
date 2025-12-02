# ML Song Recommendation System

A machine learning project that analyzes audio files to create a content-based music recommendation system using deep learning embeddings and clustering algorithms.

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
- Extract audio features from files in `genres_original/`
- Process and visualize the features
- Run the weighted K-Means (WKBSC) clustering algorithm
- Launch the recommendation UI

### 3. Choose Clustering Method
```bash
# K-Means (default)
python run_pipeline.py --clustering-method kmeans

# Gaussian Mixture Model
python run_pipeline.py --clustering-method gmm

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
- **Rich Audio Features**: MFCC, Mel-spectrograms, spectral descriptors
- **Interactive UI**: Browse songs, get recommendations, play audio
- **Visualization**: PCA-based 2D visualization of clusters
- **Deep Learning**: VaDE combines VAE and GMM for joint feature learning and clustering

## Module Details

### Clustering Algorithms
- **K-Means** (`src/clustering/kmeans.py`): Fast centroid-based clustering with hard assignments
- **GMM** (`src/clustering/gmm.py`): Probabilistic clustering with Gaussian Mixture Models
- **HDBSCAN** (`src/clustering/hdbscan.py`): Density-based clustering with noise detection
- **VaDE** (`src/clustering/vade.py`): Deep learning approach combining VAE and GMM
- **WKBSC** (`scripts/wkbsc.py`): Weighted K-Means with feature importance learning

### Feature Extraction
The system extracts these audio features:

**Traditional Features:**
- **MFCC** (Mel-frequency cepstral coefficients)
- **Mel-spectrogram**
- **Spectral centroid**
- **Spectral flatness**
- **Zero-crossing rate**
- **Genre** (one-hot encoded)

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
n_mels = 128     # Number of mel bands
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

### Learn Feature Weights (WKBSC)
```bash
python scripts/wkbsc.py
```

### Visualize Features
```bash
python scripts/ploting.py output/results
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
| [Feature Weights Guide](docs/FEATURE_WEIGHTS_GUIDE.md) | Guide to WKBSC feature weighting |

## License

See LICENSE file for details.

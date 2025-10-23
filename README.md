# ML Song Recommendation System

A machine learning project that analyzes audio files to create a content-based music recommendation system using clustering algorithms.

## Project Structure

```
ML-song-recommendation-system/
│
├── src/                          # Main source code
│   ├── clustering/               # Clustering algorithms
│   │   ├── kmeans.py            # K-Means clustering implementation
│   │   ├── hierarchical_clustering.py  # Hierarchical clustering
│   │   └── dbscan_clustering.py # DBSCAN clustering
│   │
│   ├── features/                 # Feature extraction modules
│   │   └── extract_features.py  # Audio feature extraction
│   │
│   ├── ui/                       # User interface modules
│   │   ├── modern_ui.py         # Main UI for recommendations
│   │   └── compare_ui.py        # UI for comparing clustering methods
│   │
│   └── data_collection/          # Data collection scripts
│       ├── playlist_audio_download.py  # Download audio from YouTube
│       └── deezer-song.py       # Deezer song data collection
│
├── scripts/                      # Utility and analysis scripts
│   ├── ploting.py               # Visualization scripts
│   ├── compare_clustering.py    # Compare clustering methods
│   └── wkbsc.py                 # Weighted K-means clustering
│
├── config/                       # Configuration files
│   ├── feature_vars.py          # Feature extraction parameters
│   ├── names.txt                # Song/playlist names
│   └── links.txt                # YouTube links
│
├── output/                       # Output files and results
│   ├── results/                 # Main results directory
│   ├── dbscan/                  # DBSCAN-specific results
│   ├── hierarchical/            # Hierarchical clustering results
│   └── spectral/                # Spectral clustering results
│
├── genres_original/              # Original audio files by genre
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── docs/                         # Documentation
│   └── README.md                # Detailed project documentation
│
├── run_pipeline.py              # Main pipeline execution script
└── LICENSE

```

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn librosa matplotlib pygame
```

### 2. Run the Complete Pipeline
```bash
python run_pipeline.py
```

This will:
- Extract audio features from files in `genres_original/`
- Process and visualize the features
- Run clustering algorithm (you'll be prompted to choose)
- Launch the recommendation UI

### 3. Run Specific Clustering Methods

**K-Means:**
```bash
python run_pipeline.py --clustering-method kmeans
```

**Hierarchical:**
```bash
python run_pipeline.py --clustering-method hierarchical
```

**DBSCAN:**
```bash
python run_pipeline.py --clustering-method dbscan
```

### 4. Skip Specific Steps
```bash
python run_pipeline.py --skip extract plot
```

## Features

- **Multiple Clustering Algorithms**: K-Means, Hierarchical, DBSCAN
- **Rich Audio Features**: MFCC, Mel-spectrograms, Spectral features
- **Interactive UI**: Browse songs, get recommendations, play audio
- **Comparison Tools**: Compare different clustering methods
- **Visualization**: PCA-based 2D visualization of clusters

## Module Details

### Clustering Algorithms
- **K-Means** (`src/clustering/kmeans.py`): Standard K-means with dynamic cluster selection
- **Hierarchical** (`src/clustering/hierarchical_clustering.py`): Agglomerative clustering with dendrograms
- **DBSCAN** (`src/clustering/dbscan_clustering.py`): Density-based clustering for finding arbitrary-shaped clusters
- **WKBSC** (`scripts/wkbsc.py`): Weighted K-means with feature importance learning

### Feature Extraction
The system extracts these audio features:
- **MFCC** (Mel-frequency cepstral coefficients)
- **Mel-spectrogram**
- **Spectral centroid**
- **Spectral flatness**
- **Zero-crossing rate**
- **Genre** (one-hot encoded)

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

### Run Specific Clustering Algorithm
```bash
python src/clustering/kmeans.py
python src/clustering/hierarchical_clustering.py
python src/clustering/dbscan_clustering.py
```

### Compare Clustering Methods
```bash
python scripts/compare_clustering.py
python src/ui/compare_ui.py
```

### Visualize Features
```bash
python scripts/ploting.py output/results
```

## Output Files

- `output/results/`: Extracted features (.npy files)
- `output/audio_clustering_results.csv`: K-means clustering results
- `output/dbscan/dbscan_clustering_results.csv`: DBSCAN results
- `output/hierarchical/hierarchical_clustering_results.csv`: Hierarchical results
- `output/results/clustering_comparison.md`: Comparison report

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- librosa
- matplotlib
- pygame
- tkinter (usually comes with Python)

## License

See LICENSE file for details.

## Documentation

For detailed documentation, see `docs/README.md`.

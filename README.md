# ML Song Recommendation System

A machine learning project that analyzes audio files to create a content-based music recommendation system using clustering algorithms.

## Project Structure

```
ML-song-recommendation-system/
│
├── src/                          # Main source code
│   ├── clustering/               # Clustering algorithms
│   │   ├── kmeans.py            # K-Means clustering implementation
│   │
│   ├── features/                 # Feature extraction modules
│   │   └── extract_features.py  # Audio feature extraction
│   │
│   ├── ui/                       # User interface modules
│   │   └── modern_ui.py         # Main UI for recommendations
│   │
│   └── data_collection/          # Data collection scripts
│       ├── playlist_audio_download.py  # Download audio from YouTube
│       └── deezer-song.py       # Deezer song data collection
│
├── scripts/                      # Utility and analysis scripts
│   ├── ploting.py               # Visualization scripts
│   └── wkbsc.py                 # Weighted K-means clustering
│
├── config/                       # Configuration files
│   ├── feature_vars.py          # Feature extraction parameters
│   ├── names.txt                # Song/playlist names
│   └── links.txt                # YouTube links
│
├── output/                       # Output files and results
│   ├── results/                 # Main results directory
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
- Run the weighted K-Means (WKBSC) clustering algorithm
- Launch the recommendation UI

### 3. Run K-Means Directly
```bash
python run_pipeline.py --clustering-method kmeans
```

### 4. Skip Specific Steps
```bash
python run_pipeline.py --skip extract plot
```

## Features

- **Weighted K-Means**: K-Means enhanced with WKBSC feature weighting
- **Rich Audio Features**: MFCC, Mel-spectrograms, spectral descriptors
- **Interactive UI**: Browse songs, get recommendations, play audio
- **Visualization**: PCA-based 2D visualization of clusters

## Module Details

### Clustering Algorithms
- **K-Means** (`src/clustering/kmeans.py`): K-Means clustering with PCA for visualization
- **WKBSC** (`scripts/wkbsc.py`): Weighted K-Means with feature importance learning

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

### Run K-Means Directly
```bash
python src/clustering/kmeans.py
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

- `output/results/`: Extracted features (.npy files)
- `output/audio_clustering_results.csv`: K-Means clustering results
- `output/results/wkbsc_feature_weights.npy`: Learned feature weights

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

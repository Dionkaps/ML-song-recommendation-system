# ML Song Recommendation System Documentation

This document provides a detailed overview of the ML Song Recommendation System project, explaining each component, their functionality, and how they interact to create a music recommendation system based on audio feature analysis.

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Components](#components)
   - [Audio Download Module](#audio-download-module)
   - [Feature Extraction Module](#feature-extraction-module)
   - [Feature Visualization Module](#feature-visualization-module)
   - [Clustering & Recommendation Module](#clustering--recommendation-module)
4. [Pipeline Execution](#pipeline-execution)
5. [Technical Details](#technical-details)
6. [Usage](#usage)

## Project Overview

The ML Song Recommendation System is a machine learning project that analyzes audio files to create a content-based music recommendation system. It processes audio features from songs, applies machine learning techniques to identify similar songs, and provides a graphical user interface to explore and receive recommendations.

## System Architecture

The system follows a modular pipeline architecture with these major steps:

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────────┐     ┌────────────────┐     ┌──────────────────────┐
│  Download    │     │  Extract Audio  │     │  Process Features  │     │  Visualize     │     │  Cluster & Generate  │
│  Audio Files │────>│  Features       │────>│  (Scale, Clean)    │────>│  Features      │────>│  Recommendations     │
└──────────────┘     └─────────────────┘     └────────────────────┘     └────────────────┘     └──────────────────────┘
```

## Components

### Audio Download Module

**File:** `playlist_audio_download.py`

This module handles downloading audio files from YouTube links or playlists.

**Key Functions:**
- `download_videos()`: Downloads individual videos from URLs
- `download_from_links()`: Processes a file containing YouTube URLs or playlists
- `main()`: Entry point for the module

**Process Flow:**

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Read Links from  │     │  Process Each     │     │  Download Audio   │     │  Convert to  │
│  links.txt        │────>│  URL/Playlist     │────>│  Using yt-dlp     │────>│  WAV Format  │
└───────────────────┘     └───────────────────┘     └───────────────────┘     └──────────────┘
```

### Feature Extraction Module

**Files:** 
- `extract_features.py`: Main extraction logic
- `feature_vars.py`: Constants for feature extraction parameters

This module extracts audio features from the downloaded WAV files using librosa.

**Key Functions:**
- `extract_mfcc()`: Extracts Mel-Frequency Cepstral Coefficients
- `extract_melspectrogram()`: Extracts Mel-Spectrogram
- `extract_spectral_centroid()`: Extracts Spectral Centroid
- `extract_spectral_flatness()`: Extracts Spectral Flatness
- `extract_zero_crossing_rate()`: Extracts Zero Crossing Rate
- `process_file()`: Processes a single audio file
- `run_feature_extraction()`: Processes all audio files in parallel

**Process Flow:**

```
┌──────────────┐     ┌───────────────┐     ┌────────────────────────────┐     ┌───────────────────┐
│  Load Audio  │     │  Extract      │     │  Save Features as .npy     │     │  Process Next     │
│  WAV File    │────>│  Features     │────>│  Files in results/ folder  │────>│  File (Parallel)  │
└──────────────┘     └───────────────┘     └────────────────────────────┘     └───────────────────┘
```

**Audio Features Extracted:**

```
┌──────────────────────────────┐
│ Audio Features               │
├──────────────────────────────┤
│ ┌────────────────────────┐   │
│ │ MFCC (Timbre)          │   │
│ └────────────────────────┘   │
│                              │
│ ┌────────────────────────┐   │
│ │ Mel-Spectrogram        │   │
│ │ (Frequency Analysis)   │   │
│ └────────────────────────┘   │
│                              │
│ ┌────────────────────────┐   │
│ │ Spectral Centroid      │   │
│ │ (Brightness)           │   │
│ └────────────────────────┘   │
│                              │
│ ┌────────────────────────┐   │
│ │ Spectral Flatness      │   │
│ │ (Tonality vs. Noise)   │   │
│ └────────────────────────┘   │
│                              │
│ ┌────────────────────────┐   │
│ │ Zero Crossing Rate     │   │
│ │ (Percussiveness)       │   │
│ └────────────────────────┘   │
└──────────────────────────────┘
```

### Feature Visualization Module

**File:** `ploting.py`

This module visualizes the extracted audio features using matplotlib.

**Key Functions:**
- `plot_feature()`: Creates plots for specific feature types
- `run_plotting()`: Generates plots for all features of all songs
- `main()`: Entry point with command-line argument handling

**Process Flow:**

```
┌───────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│  Load Audio and   │     │  Generate Individual  │     │  Save Visualization  │
│  Feature Files    │────>│  Feature Plots        │────>│  as PNG Files        │
└───────────────────┘     └───────────────────────┘     └──────────────────────┘
```

### Clustering & Recommendation Module

**File:** `kmeans.py`

This module performs clustering on the processed features and provides a recommendation system with a graphical user interface.

**Key Functions:**
- `build_group_weights()`: Creates feature group weights for balanced clustering
- `run_kmeans_clustering()`: Performs K-means clustering on audio features
- `launch_ui()`: Launches the graphical user interface for recommendations

**Process Flow:**

```
┌───────────────────┐     ┌──────────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  Load Processed   │     │  Apply PCA for       │     │  Perform K-means  │     │  Save Results to     │
│  Feature Files    │────>│  Dimensionality      │────>│  Clustering       │────>│  CSV File            │
└───────────────────┘     └──────────────────────┘     └───────────────────┘     └──────────────────────┘
                                                                                            │
                                                                                            ▼
                                                                              ┌──────────────────────────┐
                                                                              │  Launch Recommendation   │
                                                                              │  UI with Visualization  │
                                                                              └──────────────────────────┘
```

**User Interface Components:**

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Audio Recommendation System                              │
├─────────────────────────────┬──────────────────────────────────────────────────────┤
│                             │                                                      │
│  🔍 Search Songs:           │                  Cluster Map                         │
│  [                    ]     │                                                      │
│                             │        [PCA-based Visualization of Songs             │
│  🎶 All Songs:              │         with Clusters in Different Colors]           │
│  ┌───────────────────────┐  │                                                      │
│  │                       │  │                                                      │
│  │  Song 1               │  │                                                      │
│  │  Song 2               │  │                                                      │
│  │  Song 3               │  │                                                      │
│  │  ...                  │  ├──────────────────────────────────────────────────────┤
│  │                       │  │                                                      │
│  │                       │  │  Recommendations                                     │
│  │                       │  │  ┌─────────────────────────────────────────────────┐ │
│  │                       │  │  │ Similar Song 1                                  │ │
│  │                       │  │  │ Similar Song 2                                  │ │
│  │                       │  │  │ Similar Song 3                                  │ │
│  │                       │  │  │ ...                                             │ │
│  │                       │  │  └─────────────────────────────────────────────────┘ │
│  └───────────────────────┘  │                                                      │
│                             │                                                      │
└─────────────────────────────┴──────────────────────────────────────────────────────┘
```

## Pipeline Execution

**File:** `run_pipeline.py`

This module orchestrates the entire pipeline execution, allowing for specific steps to be skipped if desired. It also lets you choose between different clustering algorithms.

**Process Flow:**

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    run_pipeline.py                                           │
├──────────────┬───────────────────┬────────────────────┬────────────────────┬────────────────┤
│  Download    │  Extract          │  Process           │  Generate          │  Cluster &     │
│  Audio Files │  Audio Features   │  Features          │  Visualizations    │  Recommend     │
├──────────────┼───────────────────┼────────────────────┼────────────────────┼────────────────┤
│  playlist_   │  extract_         │  process_          │  ploting.py        │  kmeans.py     │
│  audio_      │  features.py      │  features.py       │                    │                │
│  download.py │                   │                    │                    │                │
└──────────────┴───────────────────┴────────────────────┴────────────────────┴────────────────┘
```

## Technical Details

### Feature Extraction Parameters

The `feature_vars.py` file defines the following constants:

- `n_mfcc = 13`: Number of MFCC coefficients
- `n_fft = 2048`: FFT window size
- `hop_length = 512`: Hop length for frame-level analysis 
- `n_mels = 128`: Number of Mel bands for spectrogram

### Features Explained

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Represents timbre and tonal content
   - Captures spectral shape in a compact form
   - Highly useful for genre classification

2. **Mel-Spectrogram**
   - Frequency representation adjusted to human hearing perception
   - Shows how frequency content changes over time
   - Captures overall sonic texture

3. **Spectral Centroid**
   - Represents the "brightness" of sound
   - Higher values indicate more high-frequency content
   - Useful for differentiating between "bright" vs. "warm" sounds

4. **Spectral Flatness**
   - Measures the tonality vs. noise-like quality of sound
   - Distinguishes between harmonic content and noise
   - Higher values indicate noise-like qualities

5. **Zero Crossing Rate**
   - Measures how often the signal changes from positive to negative
   - Higher for percussive sounds and consonants
   - Useful for rhythm analysis and voice/music discrimination

### Clustering Approaches

The system supports multiple clustering algorithms:

#### K-means Clustering
- Groups songs with similar audio characteristics
- Uses weighted feature combination (WKBSC algorithm)
- Supports dynamic cluster number selection using silhouette score

#### Hierarchical Clustering
- Creates a hierarchy of clusters using agglomerative approach
- Supports different linkage methods (ward, complete, average, single)
- Offers various distance metrics (euclidean, cityblock, cosine, correlation)

## Usage

1. **Add YouTube Links**
   - Create a `links.txt` file with YouTube video or playlist URLs

2. **Run Complete Pipeline**
   ```
   python run_pipeline.py
   ```
   
   - Choose clustering algorithm:
   ```
   python run_pipeline.py --cluster-algorithm hierarchical
   ```

3. **Run Specific Steps**
   - Skip certain steps:
   ```
   python run_pipeline.py --skip download extract
   ```

4. **Run Individual Components**
   - Download audio: `python playlist_audio_download.py`
   - Extract features: `python extract_features.py`
   - Create visualizations: `python ploting.py results`
   - Cluster and recommend: 
     - K-means: `python kmeans.py`
     - Hierarchical: `python hierarchical.py`
     - Advanced options: `python cluster.py --algorithm hierarchical --linkage-method ward --n-clusters 5`

5. **View Recommendations**
   - After running `kmeans.py`, a graphical UI will open
   - Select a song to see similar recommendations
   - Visualize song relationships in the PCA plot

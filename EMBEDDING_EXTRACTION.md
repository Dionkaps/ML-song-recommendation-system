# Audio Embedding Extraction

This module extracts audio embeddings using four state-of-the-art models:

## Models

1. **OpenL3** - Audio embeddings trained on AudioSet
2. **CREPE** - Pitch tracking using deep learning
3. **madmom** - Beat activation features using RNNs
4. **MERT** - Music understanding model (MERT-v1-95M)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `madmom`, install from GitHub:
```bash
pip install --upgrade git+https://github.com/CPJKU/madmom.git
```

## Usage

### Basic Usage

Extract all embeddings from all audio files:

```bash
python run_extraction.py
```

### Advanced Options

```bash
python run_extraction.py \
    --audio_dir audio_files \
    --output_base output/embeddings \
    --models openl3 crepe madmom mert \
    --skip_existing \
    --verbose
```

#### Arguments:

- `--audio_dir`: Directory containing audio files (default: `audio_files`)
- `--output_base`: Output directory for embeddings (default: `output/embeddings`)
- `--models`: Which models to use (default: all four)
  - Options: `openl3`, `crepe`, `madmom`, `mert`
  - Example: `--models openl3 mert` (only use OpenL3 and MERT)
- `--skip_existing`: Skip files that already have embeddings (useful for resuming)
- `--verbose`: Print detailed output including embedding shapes

### Examples

Extract only OpenL3 and MERT embeddings:
```bash
python run_extraction.py --models openl3 mert
```

Resume interrupted extraction (skip already processed files):
```bash
python run_extraction.py --skip_existing
```

Process audio from a different directory:
```bash
python run_extraction.py --audio_dir /path/to/audio --output_base /path/to/output
```

## Output Format

All embeddings are saved as NumPy arrays (`.npy` files) in the corresponding model subdirectory:

```
output/embeddings/
├── openl3/
│   ├── song1.npy  # Shape: (T, 512) - T time frames, 512-dim embeddings
│   └── song2.npy
├── crepe/
│   ├── song1.npy  # Shape: (T, 2) - frequency and confidence per frame
│   └── song2.npy
├── madmom/
│   ├── song1.npy  # Shape: (T,) - beat activation function
│   └── song2.npy
└── mert/
    ├── song1.npy  # Shape: (T, 768) - T time frames, 768-dim embeddings
    └── song2.npy
```

## Performance Optimizations

The implementation includes several key optimizations:

1. **Model Caching**: Models are loaded once and reused for all files
   - Previously, MERT was reloaded for every file (huge overhead!)
   - Now all models are cached in memory
   
2. **Progress Tracking**: Shows elapsed time and estimated remaining time
   
3. **Resume Support**: Use `--skip_existing` to skip already processed files
   
4. **Better Error Handling**: Failed extractions don't stop the entire process
   
5. **Success/Failure Reporting**: Summary statistics at the end

## Expected Processing Time

For 869 audio files (30-second clips):
- **OpenL3**: ~2-3 seconds per file
- **CREPE**: ~1-2 seconds per file  
- **madmom**: ~1-2 seconds per file
- **MERT**: ~3-5 seconds per file (CPU) or ~1 second (GPU)

**Total estimated time**: 
- CPU: ~2-3 hours for all 869 files
- GPU: ~1-1.5 hours for all 869 files

## Technical Details

### OpenL3
- Input sample rate: 48 kHz
- Model: Music content, mel-256 spectrogram
- Embedding size: 512-dimensional
- Frame rate: 10 Hz (0.1s hop size)

### CREPE
- Input sample rate: 16 kHz
- Output: Pitch frequency (Hz) + confidence per frame
- Step size: 10ms
- Uses Viterbi smoothing for better pitch tracking

### madmom
- Processes audio directly from file
- RNN-based beat activation function
- Output rate: 100 fps (10ms frames)
- Values represent beat occurrence probability

### MERT
- Input sample rate: 24 kHz
- Model: m-a-p/MERT-v1-95M
- Embedding size: 768-dimensional
- Frame rate: ~75 fps (features per second)
- GPU acceleration supported

## Troubleshooting

### madmom Installation Issues
```bash
# If pip install madmom fails, try:
pip install --upgrade git+https://github.com/CPJKU/madmom.git
```

### CUDA Out of Memory (MERT)
If you get CUDA OOM errors with MERT:
```bash
# Process in smaller batches or use CPU only
CUDA_VISIBLE_DEVICES="" python run_extraction.py --models openl3 crepe madmom
# Then run MERT separately
python run_extraction.py --models mert
```

### Progress Tracking
The script shows:
- Current file being processed
- Success/failure status for each model
- Elapsed time and estimated remaining time
- Final summary with success/failure counts

## Loading Embeddings

```python
import numpy as np

# Load OpenL3 embeddings
openl3_emb = np.load('output/embeddings/openl3/song1.npy')
print(f"OpenL3 shape: {openl3_emb.shape}")  # (T, 512)

# Load CREPE features
crepe_features = np.load('output/embeddings/crepe/song1.npy')
print(f"CREPE shape: {crepe_features.shape}")  # (T, 2)
print(f"Pitch (Hz): {crepe_features[:, 0]}")
print(f"Confidence: {crepe_features[:, 1]}")

# Load madmom activations
madmom_act = np.load('output/embeddings/madmom/song1.npy')
print(f"madmom shape: {madmom_act.shape}")  # (T,)

# Load MERT embeddings
mert_emb = np.load('output/embeddings/mert/song1.npy')
print(f"MERT shape: {mert_emb.shape}")  # (T, 768)
```

## Next Steps

After extracting embeddings, you can:
1. Use them as features for music recommendation
2. Train downstream classifiers (genre, mood, etc.)
3. Compute similarity between songs
4. Visualize with t-SNE/UMAP
5. Build a music retrieval system

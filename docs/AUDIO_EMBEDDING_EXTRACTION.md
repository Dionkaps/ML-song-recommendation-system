# Audio Embedding Extraction Guide

This guide explains how to set up and run the audio feature extraction pipeline using three state-of-the-art pretrained models: **EnCodecMAE**, **MusiCNN**, and **MERT**.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
   - [Creating the EnCodecMAE Environment (Python 3.12)](#creating-the-encodecmae-environment-python-312)
   - [Creating the MusiCNN Environment (Python 3.7)](#creating-the-musicnn-environment-python-37)
4. [Model Installation](#model-installation)
   - [EnCodecMAE Installation](#encodecmae-installation)
   - [MusiCNN Installation](#musicnn-installation)
   - [MERT Installation](#mert-installation)
5. [Running Feature Extraction](#running-feature-extraction)
   - [Quick Start](#quick-start)
   - [EnCodecMAE Extraction](#encodecmae-extraction)
   - [MERT Extraction](#mert-extraction)
   - [MusiCNN Extraction](#musicnn-extraction)
6. [Output Format](#output-format)
7. [Troubleshooting](#troubleshooting)
8. [Model Details](#model-details)

---

## Overview

This pipeline extracts audio embeddings (dense vector representations) from music files using three complementary deep learning models:

| Model | Architecture | Embedding Dim | Python Version | Best For |
|-------|-------------|---------------|----------------|----------|
| **EnCodecMAE** | Masked Autoencoder | 768 | 3.12 | General audio understanding |
| **MERT** | Transformer | 768 | 3.12 | Music-specific features |
| **MusiCNN** | CNN | 753 | 3.7 | Music tagging/classification |

**Why multiple models?** Each model captures different aspects of audio:
- **EnCodecMAE**: Self-supervised learning on audio codecs, excellent for reconstruction-based features
- **MERT**: Trained specifically on music, captures harmonic and rhythmic patterns
- **MusiCNN**: Trained on music tagging datasets, captures genre/mood characteristics

---

## Prerequisites

### Required Software

1. **Python 3.12** (for EnCodecMAE and MERT)
2. **Python 3.7** (for MusiCNN - required due to numpy<1.17 dependency)
3. **Git** (for cloning repositories)
4. **Windows** (this guide uses PowerShell commands; adapt for Linux/Mac)

### Verify Python Installations

```powershell
# Check available Python versions
py --list

# Expected output should include:
#  -V:3.12  Python 3.12.x
#  -V:3.7   Python 3.7.x
```

If you don't have these Python versions:
- Download Python 3.12 from [python.org](https://www.python.org/downloads/)
- Download Python 3.7 from [python.org/downloads/release/python-379/](https://www.python.org/downloads/release/python-379/)

> ⚠️ **Important**: Install Python versions using the official installer, not conda or pyenv, to ensure the `py` launcher works correctly.

---

## Environment Setup

We need **two separate virtual environments** because MusiCNN requires legacy dependencies (numpy<1.17) that are incompatible with modern packages.

### Creating the EnCodecMAE Environment (Python 3.12)

This environment will be used for both EnCodecMAE and MERT.

```powershell
# Navigate to your project directory
cd C:\path\to\your\project

# Create virtual environment with Python 3.12
py -3.12 -m venv .venv-encodecmae

# Activate the environment
.\.venv-encodecmae\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Verify Python version
python --version
# Should output: Python 3.12.x
```

### Creating the MusiCNN Environment (Python 3.7)

```powershell
# Create virtual environment with Python 3.7
py -3.7 -m venv .venv-musicnn

# Activate the environment
.\.venv-musicnn\Scripts\Activate.ps1

# Upgrade pip (use a compatible version for Python 3.7)
python -m pip install --upgrade pip

# Verify Python version
python --version
# Should output: Python 3.7.x
```

---

## Model Installation

### EnCodecMAE Installation

EnCodecMAE is a self-supervised audio encoder based on the EnCodec neural codec.

```powershell
# Activate the EnCodecMAE environment
.\.venv-encodecmae\Scripts\Activate.ps1

# Clone the EnCodecMAE repository
git clone https://github.com/habla-liaa/encodecmae.git

# Install EnCodecMAE in editable mode
pip install -e ./encodecmae

# Install additional dependencies
pip install librosa numpy
```

#### Verify EnCodecMAE Installation

```powershell
# IMPORTANT: Run from the encodecmae folder to avoid import conflicts
cd encodecmae

python -c "from encodecmae import load_model; print('EnCodecMAE OK')"

# Return to project root
cd ..
```

> ⚠️ **Import Conflict Warning**: The local `encodecmae` folder can shadow the installed package. Always `cd encodecmae` before importing, or run scripts from within that folder.

### MusiCNN Installation

MusiCNN is a music audio tagging model trained on MagnaTagATune and Million Song Dataset.

```powershell
# Activate the MusiCNN environment
.\.venv-musicnn\Scripts\Activate.ps1

# Install TensorFlow 2.7.4 (compatible with Python 3.7)
pip install tensorflow==2.7.4 --timeout 300

# Install librosa (audio processing)
pip install librosa==0.8.1

# Install numpy (MUST be < 1.17 for musicnn_keras)
pip install "numpy>=1.14.5,<1.17"

# Install musicnn_keras
pip install musicnn-keras
```

#### Download MusiCNN Model Weights

The `musicnn-keras` pip package does NOT include the pretrained weights. You must download them separately:

```powershell
# Clone the full repository to get the model weights
git clone https://github.com/Quint-e/musicnn_keras.git musicnn_keras_temp

# Create the expected directory structure
New-Item -ItemType Directory -Force -Path "musicnn_keras\keras_checkpoints"

# Copy the model weights
Copy-Item "musicnn_keras_temp\musicnn_keras\keras_checkpoints\*" -Destination "musicnn_keras\keras_checkpoints\" -Force

# Clean up the temporary clone
Remove-Item -Recurse -Force musicnn_keras_temp
```

The model weights should now be in `./musicnn_keras/keras_checkpoints/`:
- `MTT_musicnn.h5` - MagnaTagATune trained MusiCNN
- `MTT_vgg.h5` - MagnaTagATune trained VGG
- `MSD_musicnn.h5` - Million Song Dataset trained MusiCNN
- `MSD_vgg.h5` - Million Song Dataset trained VGG
- `MSD_musicnn_big.h5` - Larger MSD model

#### Verify MusiCNN Installation

```powershell
# Activate MusiCNN environment
.\.venv-musicnn\Scripts\Activate.ps1

python -c "from musicnn_keras.extractor import extractor; print('MusiCNN OK')"
```

### MERT Installation

MERT (Music undERstanding Transformer) uses the HuggingFace transformers library.

```powershell
# Activate the EnCodecMAE environment (MERT uses the same env)
.\.venv-encodecmae\Scripts\Activate.ps1

# Install transformers
pip install transformers

# Install torch if not already installed
pip install torch
```

#### Verify MERT Installation

```powershell
python -c "from transformers import AutoModel; print('MERT OK')"
```

The MERT model weights will be downloaded automatically from HuggingFace on first use.

---

## Running Feature Extraction

### Quick Start

Use the orchestrator script to see all available commands:

```powershell
python run_extraction.py --help
python run_extraction.py --status  # Check existing embeddings
```

### EnCodecMAE Extraction

```powershell
# Activate environment
.\.venv-encodecmae\Scripts\Activate.ps1

# IMPORTANT: Change to encodecmae folder to avoid import conflicts
cd encodecmae

# Run extraction
python ../scripts/extract_encodecmae.py `
    --audio_dir ../audio_files `
    --output_dir ../output/embeddings/encodecmae `
    --skip_existing `
    --verbose

# Return to project root
cd ..
```

#### EnCodecMAE Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio_dir` | Directory containing audio files | `audio_files` |
| `--output_dir` | Directory to save embeddings | `output/embeddings/encodecmae` |
| `--model` | Model variant | `mel256-ec-base_st` |
| `--device` | `cpu` or `cuda:0` | `cpu` |
| `--skip_existing` | Skip files with existing embeddings | False |
| `--verbose` | Print detailed output | False |
| `--limit` | Process only N files | None (all) |

### MERT Extraction

```powershell
# Activate environment
.\.venv-encodecmae\Scripts\Activate.ps1

# Run extraction (can run from project root)
python scripts/extract_mert.py `
    --audio_dir audio_files `
    --output_dir output/embeddings/mert `
    --skip_existing `
    --verbose
```

#### MERT Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio_dir` | Directory containing audio files | `audio_files` |
| `--output_dir` | Directory to save embeddings | `output/embeddings/mert` |
| `--model` | Model variant | `m-a-p/MERT-v1-95M` |
| `--skip_existing` | Skip files with existing embeddings | False |
| `--verbose` | Print detailed output | False |
| `--limit` | Process only N files | None (all) |
| `--full_sequence` | Keep temporal sequence (don't mean-pool) | False |

### MusiCNN Extraction

```powershell
# Activate environment
.\.venv-musicnn\Scripts\Activate.ps1

# Run extraction
python scripts/extract_musicnn.py `
    --audio_dir audio_files `
    --output_dir output/embeddings/musicnn `
    --skip_existing `
    --verbose
```

#### MusiCNN Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio_dir` | Directory containing audio files | `audio_files` |
| `--output_dir` | Directory to save embeddings | `output/embeddings/musicnn` |
| `--model` | Model variant (see below) | `MTT_musicnn` |
| `--skip_existing` | Skip files with existing embeddings | False |
| `--verbose` | Print detailed output | False |
| `--limit` | Process only N files | None (all) |

**Available MusiCNN Models:**
- `MTT_musicnn` - MagnaTagATune trained MusiCNN (recommended)
- `MTT_vgg` - MagnaTagATune trained VGG-like model
- `MSD_musicnn` - Million Song Dataset trained MusiCNN
- `MSD_vgg` - Million Song Dataset trained VGG-like model

---

## Output Format

### Directory Structure

```
output/
└── embeddings/
    ├── encodecmae/
    │   ├── Song Name 1.npy
    │   ├── Song Name 2.npy
    │   └── ...
    ├── mert/
    │   ├── Song Name 1.npy
    │   ├── Song Name 2.npy
    │   └── ...
    └── musicnn/
        ├── Song Name 1.npy
        ├── Song Name 2.npy
        └── ...
```

### Embedding Files

Each `.npy` file contains a single numpy array representing the audio embedding:

```python
import numpy as np

# Load an embedding
embedding = np.load("output/embeddings/mert/Song Name.npy")

print(embedding.shape)
# EnCodecMAE: (768,)
# MERT: (768,)
# MusiCNN: (753,)
```

### Using the Embeddings

```python
import numpy as np
import os

def load_all_embeddings(model_dir):
    """Load all embeddings from a model's output directory."""
    embeddings = {}
    for filename in os.listdir(model_dir):
        if filename.endswith('.npy'):
            song_name = filename[:-4]  # Remove .npy
            embeddings[song_name] = np.load(os.path.join(model_dir, filename))
    return embeddings

# Load embeddings from all models
encodecmae_emb = load_all_embeddings("output/embeddings/encodecmae")
mert_emb = load_all_embeddings("output/embeddings/mert")
musicnn_emb = load_all_embeddings("output/embeddings/musicnn")

# Concatenate embeddings for a song (if using all models)
def get_combined_embedding(song_name):
    """Combine embeddings from all models for a song."""
    emb_list = []
    
    if song_name in encodecmae_emb:
        emb_list.append(encodecmae_emb[song_name])
    if song_name in mert_emb:
        emb_list.append(mert_emb[song_name])
    if song_name in musicnn_emb:
        emb_list.append(musicnn_emb[song_name])
    
    return np.concatenate(emb_list)  # Shape: (768 + 768 + 753,) = (2289,)
```

---

## Troubleshooting

### Common Issues

#### 1. "Unknown model" error in MusiCNN

**Problem:** MusiCNN cannot find the model weights.

**Solution:** Ensure the model weights are in `./musicnn_keras/keras_checkpoints/`:

```powershell
# Check if weights exist
Get-ChildItem musicnn_keras\keras_checkpoints\
```

If missing, download them as described in [Download MusiCNN Model Weights](#download-musicnn-model-weights).

#### 2. EnCodecMAE import error

**Problem:** `ModuleNotFoundError: No module named 'encodecmae'` or wrong module imported.

**Solution:** The local `encodecmae` folder shadows the installed package. Always run from within the `encodecmae` directory:

```powershell
cd encodecmae
python ../scripts/extract_encodecmae.py --audio_dir ../audio_files ...
```

#### 3. TensorFlow CUDA warnings

**Problem:** Warnings about missing CUDA libraries.

**Solution:** These are harmless if you're using CPU. To suppress:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
```

#### 4. numpy version conflicts

**Problem:** `numpy>=1.17` conflicts with MusiCNN.

**Solution:** This is why we use separate environments. MusiCNN requires `numpy<1.17`, while modern packages require newer versions. Never mix these in one environment.

#### 5. Out of memory errors

**Problem:** System runs out of RAM processing large files.

**Solution:** 
- Process files in batches using `--limit`
- Close other applications
- Consider using GPU if available (`--device cuda:0`)

#### 6. MP3 ID3 tag warnings

**Problem:** Warnings like `ID3v2: unrealistic small tag length 0`.

**Solution:** These are harmless warnings from the audio decoder and can be ignored. The audio is still processed correctly.

### Checking Environment Status

```powershell
# Check which Python is active
python --version

# List installed packages
pip list

# Check if in virtual environment
echo $env:VIRTUAL_ENV
```

---

## Model Details

### EnCodecMAE

**Paper:** [EnCodecMAE: Towards Unified Audio Understanding](https://arxiv.org/abs/2309.07391)

**Architecture:**
- Based on Meta's EnCodec neural audio codec
- Masked Autoencoder (MAE) trained on audio reconstruction
- Encoder produces 768-dimensional embeddings

**Input Requirements:**
- Sample rate: 24 kHz
- Mono audio
- Any length (automatically batched)

**Available Models:**
- `mel256-ec-base_st` (default) - Base model with semantic tokens
- `mel256-ec-large_st` - Larger model variant

### MERT

**Paper:** [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107)

**Architecture:**
- Transformer-based (similar to HuBERT/wav2vec2)
- Pre-trained on 160K hours of music
- 95M or 330M parameters

**Input Requirements:**
- Sample rate: 24 kHz
- Mono audio
- Any length (automatically chunked)

**Available Models:**
- `m-a-p/MERT-v1-95M` (default) - 95M parameter model
- `m-a-p/MERT-v1-330M` - 330M parameter model (better quality, slower)

### MusiCNN

**Paper:** [End-to-end Learning for Music Audio Tagging at Scale](https://arxiv.org/abs/1711.02520)

**Architecture:**
- CNN with musically-motivated filter shapes
- Trained on music tagging datasets
- Outputs tag probabilities + penultimate layer features

**Input Requirements:**
- Sample rate: 16 kHz (handled automatically by librosa)
- Any length (processed in 3-second patches)

**Available Models:**
- `MTT_musicnn` - Trained on MagnaTagATune (50 tags)
- `MSD_musicnn` - Trained on Million Song Dataset (50 tags)
- `MTT_vgg` / `MSD_vgg` - VGG-like baseline models

---

## Performance Tips

### GPU Acceleration

For EnCodecMAE and MERT:

```powershell
# Use GPU if available
python scripts/extract_mert.py --device cuda:0 ...
python scripts/extract_encodecmae.py --device cuda:0 ...
```

### Batch Processing

For large collections, use `--skip_existing` to resume interrupted extractions:

```powershell
# Process in batches, resuming if interrupted
python scripts/extract_mert.py --skip_existing --limit 1000 ...
# Run again to continue
python scripts/extract_mert.py --skip_existing --limit 1000 ...
```

### Parallel Processing

Run different models in parallel (in separate terminals):

```powershell
# Terminal 1 (EnCodecMAE env)
.\.venv-encodecmae\Scripts\Activate.ps1
cd encodecmae
python ../scripts/extract_encodecmae.py ...

# Terminal 2 (EnCodecMAE env)
.\.venv-encodecmae\Scripts\Activate.ps1
python scripts/extract_mert.py ...

# Terminal 3 (MusiCNN env)
.\.venv-musicnn\Scripts\Activate.ps1
python scripts/extract_musicnn.py ...
```

---

## References

- **EnCodecMAE**: https://github.com/habla-liaa/encodecmae
- **MusiCNN (Keras)**: https://github.com/Quint-e/musicnn_keras
- **MusiCNN (Original)**: https://github.com/jordipons/musicnn
- **MERT**: https://huggingface.co/m-a-p/MERT-v1-95M

---

## License

This documentation and the extraction scripts are provided as part of the ML Song Recommendation System project. The individual models have their own licenses:

- EnCodecMAE: MIT License
- MusiCNN: ISC License
- MERT: Apache 2.0 License

Please check each model's repository for specific licensing terms.

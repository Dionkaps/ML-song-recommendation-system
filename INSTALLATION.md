# Complete Installation Guide

This guide covers how to set up the ML Song Recommendation System from scratch on a new machine, including all audio embedding extraction models.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Step 1: Clone the Repository](#step-1-clone-the-repository)
3. [Step 2: Install Python Versions](#step-2-install-python-versions)
4. [Step 3: Create Virtual Environments](#step-3-create-virtual-environments)
5. [Step 4: Install EnCodecMAE Environment](#step-4-install-encodecmae-environment)
6. [Step 5: Install MusiCNN Environment](#step-5-install-musicnn-environment)
7. [Step 6: Download Model Weights](#step-6-download-model-weights)
8. [Step 7: Verify Installation](#step-7-verify-installation)
9. [Step 8: Add Your Audio Files](#step-8-add-your-audio-files)
10. [Step 9: Run Extraction](#step-9-run-extraction)
11. [Troubleshooting](#troubleshooting)
12. [Complete Command Reference](#complete-command-reference)
13. [Quick Verification Checklist](#quick-verification-checklist)
14. [Project Structure After Installation](#appendix-project-structure-after-installation)

---

## System Requirements

### Hardware
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for models and dependencies
- **CPU**: Any modern multi-core processor
- **GPU**: Optional (CUDA-compatible for faster processing)

### GPU Support (Optional)

If you have an NVIDIA GPU and want faster processing:

1. **Check CUDA Compatibility**: Ensure your GPU supports CUDA 11.8+
2. **Install NVIDIA Drivers**: Download from https://www.nvidia.com/Download/index.aspx
3. **Install CUDA Toolkit**: Download from https://developer.nvidia.com/cuda-downloads

When installing PyTorch, use the CUDA version:
```powershell
# Instead of: pip install torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Software
- **Operating System**: Windows 10/11 (this guide uses PowerShell commands)
- **Python 3.12**: For EnCodecMAE and MERT models
- **Python 3.7**: For MusiCNN model (required due to legacy numpy dependency)
- **Git**: For cloning repositories
- **FFmpeg**: For audio file processing (required by librosa)

### Installing FFmpeg (Required)

FFmpeg is needed by librosa to load various audio formats (mp3, m4a, etc.):

**Option 1: Using winget (Windows 11 / Windows 10 with App Installer)**
```powershell
winget install FFmpeg
```

**Option 2: Using Chocolatey**
```powershell
choco install ffmpeg
```

**Option 3: Manual Installation**
1. Download from: https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH

**Verify FFmpeg Installation:**
```powershell
ffmpeg -version
```

---

## Step 1: Clone the Repository

```powershell
# Navigate to where you want the project
cd C:\Users\YourUsername\Projects

# Clone the repository
git clone https://github.com/Dionkaps/ML-song-recommendation-system.git

# Enter the project directory
cd ML-song-recommendation-system
```

---

## Step 2: Install Python Versions

### Install Python 3.12

1. Download from: https://www.python.org/downloads/release/python-3120/
2. Run the installer
3. **✅ Check "Add Python to PATH"**
4. **✅ Check "Install py launcher"**
5. Click "Install Now"

### Install Python 3.7

1. Download from: https://www.python.org/downloads/release/python-379/
2. Run the installer
3. **✅ Check "Add Python to PATH"**
4. Click "Install Now"

### Verify Installation

```powershell
# Check py launcher can see both versions
py --list
```

Expected output:
```
 -V:3.12 *        Python 3.12
 -V:3.7           Python 3.7
```

> ⚠️ **Important**: If `py --list` doesn't show both versions, you may need to:
> - Restart your terminal
> - Reinstall Python with the "py launcher" option checked
> - Add Python to PATH manually

---

## Step 3: Create Virtual Environments

We need **two separate environments** because MusiCNN requires `numpy<1.17` which is incompatible with modern packages.

```powershell
# Make sure you're in the project directory
cd C:\Users\YourUsername\Projects\ML-song-recommendation-system

# Create environment for EnCodecMAE and MERT (Python 3.12)
py -3.12 -m venv .venv-encodecmae

# Create environment for MusiCNN (Python 3.7)
py -3.7 -m venv .venv-musicnn
```

### Verify Environments Created

```powershell
# Check folders exist
Get-ChildItem -Directory -Filter ".venv*"
```

Expected output:
```
    Directory: C:\...\ML-song-recommendation-system

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         12/5/2025   ...                    .venv-encodecmae
d-----         12/5/2025   ...                    .venv-musicnn
```

---

## Step 4: Install EnCodecMAE Environment

This environment is used for **EnCodecMAE** and **MERT** models.

### 4.1 Activate Environment

```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
```

Your prompt should change to show `(.venv-encodecmae)`.

### 4.2 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4.3 Clone EnCodecMAE Repository

```powershell
# Clone the EnCodecMAE repo (contains the model code)
git clone https://github.com/habla-liaa/encodecmae.git
```

### 4.4 Install EnCodecMAE

```powershell
# Install in editable mode
pip install -e ./encodecmae
```

### 4.5 Install Additional Dependencies

```powershell
# Install audio processing library
pip install librosa

# Install for MERT model
pip install transformers

# Install PyTorch (CPU version - smaller download)
pip install torch

# Or for GPU support (CUDA 11.8):
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4.6 Deactivate Environment

```powershell
deactivate
```

---

## Step 5: Install MusiCNN Environment

This environment is used for the **MusiCNN** model only.

### 5.1 Activate Environment

```powershell
.\.venv-musicnn\Scripts\Activate.ps1
```

### 5.2 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5.3 Install TensorFlow

```powershell
# Install TensorFlow 2.7.4 (compatible with Python 3.7)
# Use --timeout for slow connections
pip install tensorflow==2.7.4 --timeout 300
```

> ⏱️ This may take 5-10 minutes depending on your internet connection.

### 5.4 Install Audio Processing Libraries

```powershell
# Install librosa (specific version for compatibility)
pip install librosa==0.8.1

# Install numpy (MUST be < 1.17 for musicnn_keras)
pip install "numpy>=1.14.5,<1.17"
```

### 5.5 Install MusiCNN

```powershell
pip install musicnn-keras
```

### 5.6 Deactivate Environment

```powershell
deactivate
```

---

## Step 6: Download Model Weights

### 6.1 MusiCNN Weights (Required)

The `musicnn-keras` pip package does **NOT** include the pretrained model weights. You must download them manually:

```powershell
# Clone the full musicnn_keras repo to get the weights
git clone https://github.com/Quint-e/musicnn_keras.git musicnn_keras_temp

# Create the directory structure expected by the library
New-Item -ItemType Directory -Force -Path "musicnn_keras\keras_checkpoints"

# Copy the model weight files
Copy-Item "musicnn_keras_temp\musicnn_keras\keras_checkpoints\*" -Destination "musicnn_keras\keras_checkpoints\" -Force

# Clean up the temporary clone
Remove-Item -Recurse -Force musicnn_keras_temp
```

### 6.2 Verify Weights Downloaded

```powershell
Get-ChildItem "musicnn_keras\keras_checkpoints\"
```

Expected output:
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         ...                                MSD_musicnn.h5
-a----         ...                                MSD_musicnn_big.h5
-a----         ...                                MSD_vgg.h5
-a----         ...                                MTT_musicnn.h5
-a----         ...                                MTT_vgg.h5
```

### 6.3 EnCodecMAE and MERT Weights (Automatic)

These models download their weights automatically from HuggingFace on first use. No manual download needed!

---

## Step 7: Verify Installation

### 7.1 Test EnCodecMAE

```powershell
# Activate environment
.\.venv-encodecmae\Scripts\Activate.ps1

# IMPORTANT: Must run from encodecmae folder to avoid import conflicts
cd encodecmae

# Test import
python -c "from encodecmae import load_model; print('✅ EnCodecMAE OK')"

# Return to project root
cd ..

# Deactivate
deactivate
```

### 7.2 Test MERT

```powershell
# Activate environment
.\.venv-encodecmae\Scripts\Activate.ps1

# Test import
python -c "from transformers import AutoModel; print('✅ MERT OK')"

# Deactivate
deactivate
```

### 7.3 Test MusiCNN

```powershell
# Activate environment
.\.venv-musicnn\Scripts\Activate.ps1

# Test import
python -c "from musicnn_keras.extractor import extractor; print('✅ MusiCNN OK')"

# Deactivate
deactivate
```

### 7.4 Test Orchestrator Script

```powershell
# This uses your system Python, not a venv
python run_extraction.py --status
```

---

## Step 8: Add Your Audio Files

Place your audio files in the `audio_files/` directory:

```powershell
# Create the directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "audio_files"

# Copy your audio files
# Example: Copy-Item "C:\Music\*.mp3" -Destination "audio_files\"
```

**Supported formats**: `.mp3`, `.wav`, `.flac`, `.m4a`

---

## Step 9: Run Extraction

### Option A: Run All Models (Recommended)

```powershell
python run_extraction.py --run --skip_existing
```

This automatically:
1. Runs EnCodecMAE extraction
2. Runs MERT extraction  
3. Runs MusiCNN extraction

### Option B: Run Individual Models

#### EnCodecMAE Only
```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
cd encodecmae
python ../scripts/extract_encodecmae.py --audio_dir ../audio_files --output_dir ../output/embeddings/encodecmae --skip_existing --verbose
cd ..
deactivate
```

#### MERT Only
```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
python scripts/extract_mert.py --audio_dir audio_files --output_dir output/embeddings/mert --skip_existing --verbose
deactivate
```

#### MusiCNN Only
```powershell
.\.venv-musicnn\Scripts\Activate.ps1
python scripts/extract_musicnn.py --audio_dir audio_files --output_dir output/embeddings/musicnn --skip_existing --verbose
deactivate
```

### Check Progress

```powershell
python run_extraction.py --status
```

---

## Troubleshooting

### Problem: `py --list` doesn't show Python versions

**Solution**: Reinstall Python with "Install py launcher" checked, or manually add Python to PATH.

### Problem: `pip install tensorflow` times out

**Solution**: Use a longer timeout:
```powershell
pip install tensorflow==2.7.4 --timeout 600
```

### Problem: "Unknown model" error in MusiCNN

**Solution**: The model weights are missing. Re-run Step 6.1 to download them.

### Problem: EnCodecMAE import error

**Solution**: You must run EnCodecMAE scripts from inside the `encodecmae/` folder:
```powershell
cd encodecmae
python ../scripts/extract_encodecmae.py ...
```

### Problem: `numpy` version conflict

**Solution**: Never install packages in the wrong environment. MusiCNN requires `numpy<1.17`, while EnCodecMAE requires newer numpy. That's why we use separate environments.

### Problem: CUDA/GPU warnings

**Solution**: These are safe to ignore if you're using CPU. The models will still work.

### Problem: "Permission denied" when activating venv

**Solution**: Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: Out of memory

**Solution**: 
- Close other applications
- Process fewer files at once: `--limit 100`
- Use CPU instead of GPU

### Problem: "Could not load dynamic library" or audio loading errors

**Solution**: FFmpeg is not installed or not in PATH. Install FFmpeg:
```powershell
winget install FFmpeg
```
Then restart your terminal.

### Problem: "No audio backend available" or "librosa" errors

**Solution**: 
1. Make sure FFmpeg is installed
2. Try installing soundfile:
```powershell
pip install soundfile
```

### Problem: HuggingFace model download slow or fails

**Solution**: The MERT and EnCodecMAE models are downloaded from HuggingFace on first run. If download is slow:
1. Check your internet connection
2. Try using a VPN
3. Manually download from HuggingFace and cache locally

---

## Complete Command Reference

### One-Time Setup (Copy-Paste Ready)

```powershell
# ============================================
# COMPLETE SETUP SCRIPT
# Run this after cloning the repository
# ============================================

# 1. Create virtual environments
py -3.12 -m venv .venv-encodecmae
py -3.7 -m venv .venv-musicnn

# 2. Setup EnCodecMAE environment
.\.venv-encodecmae\Scripts\Activate.ps1
python -m pip install --upgrade pip
git clone https://github.com/habla-liaa/encodecmae.git
pip install -e ./encodecmae
pip install librosa transformers torch
deactivate

# 3. Setup MusiCNN environment
.\.venv-musicnn\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow==2.7.4 --timeout 300
pip install librosa==0.8.1 "numpy>=1.14.5,<1.17"
pip install musicnn-keras
deactivate

# 4. Download MusiCNN weights
git clone https://github.com/Quint-e/musicnn_keras.git musicnn_keras_temp
New-Item -ItemType Directory -Force -Path "musicnn_keras\keras_checkpoints"
Copy-Item "musicnn_keras_temp\musicnn_keras\keras_checkpoints\*" -Destination "musicnn_keras\keras_checkpoints\" -Force
Remove-Item -Recurse -Force musicnn_keras_temp

# 5. Verify installation
python run_extraction.py --status

Write-Host "✅ Installation complete!" -ForegroundColor Green
```

### Daily Usage

```powershell
# Check status
python run_extraction.py --status

# Run all extractions
python run_extraction.py --run --skip_existing

# Run with verbose output
python run_extraction.py --run --skip_existing --verbose

# Test with limited files
python run_extraction.py --run --limit 10 --verbose
```

---

## Summary

| Component | Environment | Python | Key Dependencies |
|-----------|-------------|--------|------------------|
| EnCodecMAE | `.venv-encodecmae` | 3.12 | encodecmae, librosa, torch |
| MERT | `.venv-encodecmae` | 3.12 | transformers, torch |
| MusiCNN | `.venv-musicnn` | 3.7 | tensorflow 2.7.4, musicnn-keras, numpy<1.17 |

| Output | Location | Dimensions |
|--------|----------|------------|
| EnCodecMAE embeddings | `output/embeddings/encodecmae/` | 768 |
| MERT embeddings | `output/embeddings/mert/` | 768 |
| MusiCNN embeddings | `output/embeddings/musicnn/` | 753 |

---

## Quick Verification Checklist

After completing the installation, verify everything works:

- [ ] `py --list` shows Python 3.12 and 3.7
- [ ] `.venv-encodecmae` folder exists
- [ ] `.venv-musicnn` folder exists
- [ ] `encodecmae/` folder exists (cloned repo)
- [ ] `musicnn_keras/keras_checkpoints/` contains `.h5` files
- [ ] EnCodecMAE test passes: `python -c "from encodecmae import load_model; print('OK')"`
- [ ] MERT test passes: `python -c "from transformers import AutoModel; print('OK')"`
- [ ] MusiCNN test passes: `python -c "from musicnn_keras.extractor import extractor; print('OK')"`
- [ ] `python run_extraction.py --status` runs without errors

---

## Need Help?

- **Full Documentation**: See `docs/AUDIO_EMBEDDING_EXTRACTION.md`
- **Quick Reference**: See `docs/QUICKSTART_EMBEDDINGS.md`
- **Issues**: Open an issue on GitHub

---

## Appendix: Project Structure After Installation

After completing this guide, your project should have this structure:

```
ML-song-recommendation-system/
├── .venv-encodecmae/          # Python 3.12 virtual environment
├── .venv-musicnn/             # Python 3.7 virtual environment
├── audio_files/               # Your audio files go here
├── encodecmae/                # Cloned EnCodecMAE repository
├── musicnn_keras/
│   └── keras_checkpoints/     # MusiCNN model weights
│       ├── MSD_musicnn.h5
│       ├── MSD_musicnn_big.h5
│       ├── MSD_vgg.h5
│       ├── MTT_musicnn.h5
│       └── MTT_vgg.h5
├── output/
│   └── embeddings/
│       ├── encodecmae/        # EnCodecMAE outputs
│       ├── mert/              # MERT outputs
│       └── musicnn/           # MusiCNN outputs
├── scripts/
│   ├── extract_encodecmae.py
│   ├── extract_mert.py
│   └── extract_musicnn.py
├── run_extraction.py          # Main orchestrator script
├── INSTALLATION.md            # This file
└── requirements.txt
```

# Audio Embedding Extraction - Quick Start

Get audio embeddings from your music files in 5 minutes.

## Prerequisites

- Python 3.12 installed
- Python 3.7 installed
- Git installed

Check with: `py --list`

---

## 1. Create Environments (One-Time Setup)

```powershell
# Environment for EnCodecMAE + MERT (Python 3.12)
py -3.12 -m venv .venv-encodecmae

# Environment for MusiCNN (Python 3.7)
py -3.7 -m venv .venv-musicnn
```

---

## 2. Install Dependencies

### EnCodecMAE + MERT Environment

```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
pip install --upgrade pip
git clone https://github.com/habla-liaa/encodecmae.git
pip install -e ./encodecmae
pip install librosa transformers torch
```

### MusiCNN Environment

```powershell
.\.venv-musicnn\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow==2.7.4 --timeout 300
pip install librosa==0.8.1 "numpy>=1.14.5,<1.17" musicnn-keras

# Download model weights
git clone https://github.com/Quint-e/musicnn_keras.git musicnn_keras_temp
New-Item -ItemType Directory -Force -Path "musicnn_keras\keras_checkpoints"
Copy-Item "musicnn_keras_temp\musicnn_keras\keras_checkpoints\*" -Destination "musicnn_keras\keras_checkpoints\"
Remove-Item -Recurse -Force musicnn_keras_temp
```

---

## 3. Run Extraction

Place your audio files in `audio_files/` folder, then:

### EnCodecMAE
```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
cd encodecmae
python ../scripts/extract_encodecmae.py --audio_dir ../audio_files --output_dir ../output/embeddings/encodecmae --skip_existing
cd ..
```

### MERT
```powershell
.\.venv-encodecmae\Scripts\Activate.ps1
python scripts/extract_mert.py --audio_dir audio_files --output_dir output/embeddings/mert --skip_existing
```

### MusiCNN
```powershell
.\.venv-musicnn\Scripts\Activate.ps1
python scripts/extract_musicnn.py --audio_dir audio_files --output_dir output/embeddings/musicnn --skip_existing
```

---

## 4. Use Embeddings

```python
import numpy as np

# Load embedding
emb = np.load("output/embeddings/mert/My Song.npy")
print(emb.shape)  # (768,) for MERT/EnCodecMAE, (753,) for MusiCNN
```

---

## Output Summary

| Model | Embedding Size | Output Location |
|-------|---------------|-----------------|
| EnCodecMAE | 768 | `output/embeddings/encodecmae/` |
| MERT | 768 | `output/embeddings/mert/` |
| MusiCNN | 753 | `output/embeddings/musicnn/` |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "Unknown model" in MusiCNN | Model weights missing - re-run the git clone step |
| EnCodecMAE import error | Must run from `encodecmae/` folder |
| CUDA warnings | Safe to ignore if using CPU |

---

📖 **Full documentation:** [AUDIO_EMBEDDING_EXTRACTION.md](AUDIO_EMBEDDING_EXTRACTION.md)

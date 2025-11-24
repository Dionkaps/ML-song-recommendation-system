import os
import numpy as np
import librosa
import soundfile as sf
import openl3
import crepe
import madmom
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import tensorflow as tf

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Global model cache
_OPENL3_MODEL = None
_MERT_MODEL = None
_MERT_FEATURE_EXTRACTOR = None
_MERT_DEVICE = None
_MADMOM_PROCESSOR = None


def get_openl3_model():
    """Load and cache OpenL3 model."""
    global _OPENL3_MODEL
    if _OPENL3_MODEL is None:
        print("Loading OpenL3 model...")
        _OPENL3_MODEL = openl3.models.load_audio_embedding_model(
            input_repr="mel256", 
            content_type="music", 
            embedding_size=512
        )
    return _OPENL3_MODEL


def get_mert_model():
    """Load and cache MERT model and feature extractor."""
    global _MERT_MODEL, _MERT_FEATURE_EXTRACTOR, _MERT_DEVICE
    if _MERT_MODEL is None:
        print("Loading MERT model...")
        _MERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "m-a-p/MERT-v1-95M"
        _MERT_FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True
        )
        _MERT_MODEL = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        ).to(_MERT_DEVICE)
        _MERT_MODEL.eval()  # Set to evaluation mode
    return _MERT_MODEL, _MERT_FEATURE_EXTRACTOR, _MERT_DEVICE


def get_madmom_processor():
    """Load and cache madmom processor."""
    global _MADMOM_PROCESSOR
    if _MADMOM_PROCESSOR is None:
        print("Loading madmom processor...")
        _MADMOM_PROCESSOR = madmom.features.beats.RNNBeatProcessor()
    return _MADMOM_PROCESSOR


def extract_openl3(audio_path, output_dir, model=None, verbose=False):
    """
    Extracts OpenL3 embeddings.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save embeddings
        model: Pre-loaded OpenL3 model (optional, for efficiency)
        verbose: Print verbose output
    """
    try:
        if model is None:
            model = get_openl3_model()
        
        # Load audio at 48kHz (OpenL3 default)
        audio, sr = librosa.load(audio_path, sr=48000)
        
        # Extract embeddings with the pre-loaded model
        emb, ts = openl3.get_audio_embedding(
            audio, sr, 
            model=model,
            verbose=1 if verbose else 0
        )
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, emb)
        
        if verbose:
            print(f"Saved OpenL3 embedding for {basename} (shape: {emb.shape})")
        return True
    except Exception as e:
        print(f"Error extracting OpenL3 for {audio_path}: {e}")
        return False


def extract_crepe(audio_path, output_dir, verbose=False):
    """
    Extracts CREPE pitch estimation.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save features
        verbose: Print verbose output
    """
    try:
        # CREPE expects 16kHz audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Predict pitch with viterbi smoothing
        time, frequency, confidence, activation = crepe.predict(
            audio, sr, 
            viterbi=True, 
            step_size=10,  # 10ms steps
            verbose=0
        )
        
        # Stack frequency and confidence for a richer representation
        features = np.stack([frequency, confidence], axis=1)
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, features)
        
        if verbose:
            print(f"Saved CREPE features for {basename} (shape: {features.shape})")
        return True
    except Exception as e:
        print(f"Error extracting CREPE for {audio_path}: {e}")
        return False


def extract_madmom(audio_path, output_dir, processor=None, verbose=False):
    """
    Extracts madmom beat activation features.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save features
        processor: Pre-loaded madmom processor (optional)
        verbose: Print verbose output
    """
    try:
        if processor is None:
            processor = get_madmom_processor()
        
        # Process audio file directly (madmom handles loading)
        activations = processor(audio_path)
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, activations)
        
        if verbose:
            print(f"Saved madmom features for {basename} (shape: {activations.shape})")
        return True
    except Exception as e:
        print(f"Error extracting madmom for {audio_path}: {e}")
        return False


def extract_mert(audio_path, output_dir, model=None, feature_extractor=None, device=None, verbose=False):
    """
    Extracts MERT embeddings.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save embeddings
        model: Pre-loaded MERT model (optional)
        feature_extractor: Pre-loaded feature extractor (optional)
        device: torch device (optional)
        verbose: Print verbose output
    """
    try:
        if model is None or feature_extractor is None:
            model, feature_extractor, device = get_mert_model()
        
        # Load audio at 24kHz (MERT requirement)
        audio, sr = librosa.load(audio_path, sr=24000)
        
        # Prepare inputs
        inputs = feature_extractor(
            audio, 
            sampling_rate=24000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(device)
        
        # Extract features
        with torch.no_grad():
            outputs = model(input_values)
            # Get last hidden state: (batch, time, hidden_size)
            last_hidden_state = outputs.last_hidden_state
        
        # Convert to numpy
        embeddings = last_hidden_state.squeeze(0).cpu().numpy()
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, embeddings)
        
        if verbose:
            print(f"Saved MERT embeddings for {basename} (shape: {embeddings.shape})")
        return True
    except Exception as e:
        print(f"Error extracting MERT for {audio_path}: {e}")
        return False

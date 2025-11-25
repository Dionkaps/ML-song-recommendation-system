import os
import numpy as np
import librosa
import soundfile as sf
import warnings
import sys
from pathlib import Path

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings("ignore")

def create_dummy_audio(filename="dummy.wav", duration=2.0, sr=44100):
    """Create a simple sine wave audio file for testing."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(filename, audio, sr)
    return filename

def verify_openl3(audio_path):
    print("\nTesting OpenL3...")
    try:
        import openl3
        audio, sr = librosa.load(audio_path, sr=48000)
        emb, ts = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512, verbose=0)
        print(f"  ✓ OpenL3 success! Embedding shape: {emb.shape}")
        return True
    except Exception as e:
        print(f"  ✗ OpenL3 failed: {e}")
        return False

def verify_crepe(audio_path):
    print("\nTesting CREPE...")
    try:
        import crepe
        audio, sr = librosa.load(audio_path, sr=16000)
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=50, verbose=0)
        print(f"  ✓ CREPE success! Output shape: {frequency.shape}")
        return True
    except Exception as e:
        print(f"  ✗ CREPE failed: {e}")
        return False

def verify_madmom(audio_path):
    print("\nTesting madmom...")
    try:
        import madmom
        proc = madmom.features.beats.RNNBeatProcessor()
        activations = proc(audio_path)
        print(f"  ✓ madmom success! Activations shape: {activations.shape}")
        return True
    except Exception as e:
        print(f"  ✗ madmom failed: {e}")
        return False

def verify_mert(audio_path):
    print("\nTesting MERT...")
    try:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, AutoModel
        
        model_id = "m-a-p/MERT-v1-95M"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        
        audio, sr = librosa.load(audio_path, sr=24000)
        inputs = feature_extractor(audio, sampling_rate=24000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(inputs.input_values)
            last_hidden_state = outputs.last_hidden_state
            
        print(f"  ✓ MERT success! Embedding shape: {last_hidden_state.shape}")
        return True
    except Exception as e:
        print(f"  ✗ MERT failed: {e}")
        return False

def main():
    print("=== Audio Model Verification ===")
    dummy_file = create_dummy_audio()
    
    results = {
        "OpenL3": verify_openl3(dummy_file),
        "CREPE": verify_crepe(dummy_file),
        "madmom": verify_madmom(dummy_file),
        "MERT": verify_mert(dummy_file)
    }
    
    # Clean up
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
        
    print("\n=== Summary ===")
    all_passed = True
    for model, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{model}: {status}")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\nAll models are working correctly!")
    else:
        print("\nSome models failed verification.")

if __name__ == "__main__":
    main()

import os
import librosa
import numpy as np
import glob

def extract_mfcc(y, sr, n_mfcc=13):
    print(f"Extracting {n_mfcc} MFCCs")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def extract_melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    print(f"Extracting Mel-Spectrogram")
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convertion to dB
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def extract_spectral_centroid(y, sr, hop_length=512, n_fft=2048):
    print(f"Extracting Spectral Centroid")
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    return spectral_centroid

def extract_spectral_flatness(y, hop_length=512, n_fft=2048):
    print(f"Extracting Spectral Flatness")
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)
    return spectral_flatness

def extract_zero_crossing_rate(y, hop_length=512):
    print(f"Extracting Zero-Crossing Rate")
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    return zero_crossing_rate

def save_feature(feature, output_file):
    print(f"Saving feature to: {output_file}")
    np.save(output_file, feature)

def main():
    # Input and results folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio_files")
    results_dir = os.path.join(current_dir, "results")
    
    # Create results folder if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find all .wav files in the audio_files folder
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print("No .wav files found in the audio_files directory.")
        return
    
    # Parameters for feature extraction
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing file: {base_filename}.wav")
        
        output_mfcc_path = os.path.join(results_dir, f"{base_filename}_mfcc.npy")
        output_melspectrogram_path = os.path.join(results_dir, f"{base_filename}_melspectrogram.npy")
        output_spectral_centroid_path = os.path.join(results_dir, f"{base_filename}_spectral_centroid.npy")
        output_spectral_flatness_path = os.path.join(results_dir, f"{base_filename}_spectral_flatness.npy")
        output_zero_crossing_rate_path = os.path.join(results_dir, f"{base_filename}_zero_crossing_rate.npy")
        
        try:
            print(f"Loading audio file: {wav_path}")
            y, sr = librosa.load(wav_path, sr=None)  # sr=None diatirei to original sampling rate gia na mi xathei pliroforia
            print(f"Audio loaded successfully with sampling rate = {sr} Hz")
        
            # Extract MFCCs
            mfccs = extract_mfcc(y, sr, n_mfcc=n_mfcc)
            save_feature(mfccs, output_mfcc_path)
        
            # Extract Mel-Spectrogram
            mel_spectrogram = extract_melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            save_feature(mel_spectrogram, output_melspectrogram_path)
        
            # Extract Spectral Centroid
            spectral_centroid = extract_spectral_centroid(y, sr, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_centroid, output_spectral_centroid_path)
        
            # Extract Spectral Flatness
            spectral_flatness = extract_spectral_flatness(y, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_flatness, output_spectral_flatness_path)
        
            # Extract Zero-Crossing Rate
            zero_crossing_rate = extract_zero_crossing_rate(y, hop_length=hop_length)
            save_feature(zero_crossing_rate, output_zero_crossing_rate_path)
        
            print(f"Feature extraction for '{base_filename}.wav' completed successfully.")
        
        except Exception as e:
            print(f"An error occurred while processing '{base_filename}.wav': {e}")

if __name__ == "__main__":
    main()

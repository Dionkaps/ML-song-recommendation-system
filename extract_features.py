import os
import librosa
import numpy as np
import glob

def extract_mfcc(y, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def extract_melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def extract_spectral_centroid(y, sr, hop_length=512, n_fft=2048):
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

def extract_spectral_flatness(y, hop_length=512, n_fft=2048):
    return librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)

def extract_zero_crossing_rate(y, hop_length=512):
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)

def save_feature(feature, output_file):
    np.save(output_file, feature)

def run_feature_extraction(audio_dir='audio_files', results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {audio_dir}.")
        return
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nExtracting features from: {base_filename}.wav")
        output_mfcc_path = os.path.join(results_dir, f"{base_filename}_mfcc.npy")
        output_melspectrogram_path = os.path.join(results_dir, f"{base_filename}_melspectrogram.npy")
        output_spectral_centroid_path = os.path.join(results_dir, f"{base_filename}_spectral_centroid.npy")
        output_spectral_flatness_path = os.path.join(results_dir, f"{base_filename}_spectral_flatness.npy")
        output_zero_crossing_rate_path = os.path.join(results_dir, f"{base_filename}_zero_crossing_rate.npy")
        try:
            print(f"Loading audio file: {wav_path}")
            y, sr = librosa.load(wav_path, sr=None)
            print(f"Audio loaded successfully with sampling rate = {sr} Hz")
            mfccs = extract_mfcc(y, sr, n_mfcc=n_mfcc)
            save_feature(mfccs, output_mfcc_path)
            mel_spectrogram = extract_melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            save_feature(mel_spectrogram, output_melspectrogram_path)
            spectral_centroid = extract_spectral_centroid(y, sr, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_centroid, output_spectral_centroid_path)
            spectral_flatness = extract_spectral_flatness(y, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_flatness, output_spectral_flatness_path)
            zcr = extract_zero_crossing_rate(y, hop_length=hop_length)
            save_feature(zcr, output_zero_crossing_rate_path)
            print(f"Features extracted for '{base_filename}.wav'.")
        except Exception as e:
            print(f"An error occurred while processing '{base_filename}.wav': {e}")

def main():
    run_feature_extraction()

if __name__ == "__main__":
    main()

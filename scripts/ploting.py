import argparse
import glob
import os
from pathlib import Path

import librosa
from matplotlib import pyplot as plt
import numpy as np

def plot_feature(feature, y, sr, hop_length,
                 feature_type='mfcc',
                 n_fft=2048,
                 output_dir=None,
                 base_filename=None):
    plt.figure(figsize=(10, 4))
    song_name = base_filename if base_filename else "Unknown Song"

    if feature_type == 'mfcc':
        librosa.display.specshow(feature, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'MFCCs for "{song_name}"')
    elif feature_type == 'melspectrogram':
        librosa.display.specshow(feature, x_axis='time', y_axis='mel',
                                 sr=sr, hop_length=hop_length, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (dB) for "{song_name}"')
    elif feature_type == 'spectral_centroid':
        times = np.arange(len(y)) / sr
        plt.plot(times, y, alpha=0.4, label='Waveform')
        frames = range(len(feature[0]))
        t_centroid = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        centroid_hz = feature[0]
        plt.plot(t_centroid, centroid_hz, color='r', label='Spectral Centroid')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude / Frequency (Hz)')
        plt.title(f'Spectral Centroid for "{song_name}"')
        plt.legend()
    elif feature_type == 'spectral_flatness':
        frames = range(len(feature[0]))
        t_flatness = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_flatness, feature[0], color='g', label='Spectral Flatness')
        plt.xlabel('Time (s)')
        plt.ylabel('Flatness')
        plt.title(f'Spectral Flatness for "{song_name}"')
        plt.legend()
    elif feature_type == 'zero_crossing_rate':
        frames = range(len(feature[0]))
        t_zcr = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_zcr, feature[0], color='m', label='Zero-Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Rate')
        plt.title(f'Zero-Crossing Rate for "{song_name}"')
        plt.legend()
    else:
        raise ValueError("Invalid feature_type for plotting.")

    plt.tight_layout()
    if output_dir and base_filename:
        output_dir = Path(output_dir)
        plot_filename = f"{base_filename}_{feature_type}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    else:
        plt.show()

def run_plotting(audio_dir='genres_original', results_dir='output/results'):
    audio_dir_path = Path(audio_dir)
    results_dir_path = Path(results_dir)

    if not results_dir_path.exists():
        print(f"Results directory {results_dir_path} not found. Skipping plotting.")
        return

    # Get all genre directories
    genre_dirs = [d for d in glob.glob(str(audio_dir_path / "*")) if Path(d).is_dir()]
    if not genre_dirs:
        print(f"No genre directories found in {audio_dir_path}.")
        return
        
    # Collect all .wav files from all genre directories
    wav_files = []
    for genre_dir in genre_dirs:
        genre_name = os.path.basename(genre_dir)
        genre_wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
        wav_files.extend(genre_wav_files)
        print(f"Found {len(genre_wav_files)} .wav files in {genre_name} genre.")
    
    if not wav_files:
        print("No .wav files found in any genre directory.")
        return

    is_processed = results_dir_path.name == "processed_results"
    suffix = "_processed.npy" if is_processed else ".npy"

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nGenerating plots for: {base_filename}.wav")
        feature_files = {
            'mfcc': results_dir_path / f"{base_filename}_mfcc{suffix}",
            'melspectrogram': results_dir_path / f"{base_filename}_melspectrogram{suffix}",
            'spectral_centroid': results_dir_path / f"{base_filename}_spectral_centroid{suffix}",
            'spectral_flatness': results_dir_path / f"{base_filename}_spectral_flatness{suffix}",
            'zero_crossing_rate': results_dir_path / f"{base_filename}_zero_crossing_rate{suffix}"
        }

        missing_features = [ft for ft, path in feature_files.items() if not path.is_file()]
        if missing_features:
            print(f"Missing feature files for '{base_filename}.wav': {', '.join(missing_features)}. Skipping.")
            continue

        try:
            y, sr = librosa.load(wav_path, sr=None)
            print(f"Loaded '{base_filename}.wav' with sampling rate = {sr} Hz")

            mfccs = np.load(feature_files['mfcc'])
            mel_spectrogram = np.load(feature_files['melspectrogram'])
            spectral_centroid = np.load(feature_files['spectral_centroid'])
            spectral_flatness = np.load(feature_files['spectral_flatness'])
            zero_crossing_rate = np.load(feature_files['zero_crossing_rate'])

            hop_length = 512
            n_fft = 2048

            plot_feature(mfccs, y=y, sr=sr, hop_length=hop_length,
                         feature_type='mfcc', output_dir=results_dir_path,
                         base_filename=base_filename)
            plot_feature(mel_spectrogram, y=y, sr=sr, hop_length=hop_length,
                         feature_type='melspectrogram', output_dir=results_dir_path,
                         base_filename=base_filename)
            plot_feature(spectral_centroid, y=y, sr=sr, hop_length=hop_length,
                         feature_type='spectral_centroid', n_fft=n_fft,
                         output_dir=results_dir_path, base_filename=base_filename)
            plot_feature(spectral_flatness, y=y, sr=sr, hop_length=hop_length,
                         feature_type='spectral_flatness', output_dir=results_dir_path,
                         base_filename=base_filename)
            plot_feature(zero_crossing_rate, y=y, sr=sr, hop_length=hop_length,
                         feature_type='zero_crossing_rate', output_dir=results_dir_path,
                         base_filename=base_filename)

            print(f"Plots saved for '{base_filename}.wav'.")
        except Exception as e:
            print(f"Error plotting features for '{base_filename}.wav': {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot audio features from extracted data.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="output/results",
        help="Path to the folder containing feature arrays (default: output/results)"
    )
    args = parser.parse_args()
    run_plotting(results_dir=args.results_dir)

if __name__ == "__main__":
    main()

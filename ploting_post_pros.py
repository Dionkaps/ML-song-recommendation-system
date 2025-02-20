import glob
import os
import librosa
import librosa.display
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
        librosa.display.specshow(
            feature, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'MFCCs for "{song_name}"')
    elif feature_type == 'melspectrogram':
        librosa.display.specshow(
            feature, x_axis='time', y_axis='mel',
            sr=sr, hop_length=hop_length, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (dB) for "{song_name}"')
    elif feature_type == 'spectral_centroid':
        times = np.arange(len(y)) / sr
        plt.plot(times, y, alpha=0.4, label='Waveform')
        frames = range(feature.shape[1])
        t_centroid = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        centroid_hz = feature[0]
        plt.plot(t_centroid, centroid_hz, color='r', label='Spectral Centroid')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude / Frequency (Hz)')
        plt.title(f'Spectral Centroid for "{song_name}"')
        plt.legend()
    elif feature_type == 'spectral_flatness':
        frames = range(feature.shape[1])
        t_flatness = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_flatness, feature[0], color='g', label='Spectral Flatness')
        plt.xlabel('Time (s)')
        plt.ylabel('Flatness')
        plt.title(f'Spectral Flatness for "{song_name}"')
        plt.legend()
    elif feature_type == 'zero_crossing_rate':
        frames = range(feature.shape[1])
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
        plot_filename = f"{base_filename}_{feature_type}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    else:
        plt.show()


def run_plotting():
    audio_dir = 'audio_files'
    processed_dir = 'processed_results'

    # Find all processed feature files in the processed_results folder
    processed_files = glob.glob(os.path.join(processed_dir, '*_processed.npy'))
    if not processed_files:
        print("No processed files found in 'processed_results'.")
        return

    # Group feature files by base filename, e.g. "song_mfcc_processed.npy" -> base "song"
    feature_dict = {}
    for fpath in processed_files:
        fname = os.path.basename(fpath)
        # Split something like "song_mfcc_processed.npy" -> base "song", type "mfcc"
        try:
            base, feature_type, _ = fname.rsplit('_', 2)
            feature_dict.setdefault(base, {})[feature_type] = fpath
        except ValueError:
            print(f"Skipping unrecognized file format: {fname}")

    # For each base filename, check if we have all or some features
    for base_filename, feat_files in feature_dict.items():
        wav_path = os.path.join(audio_dir, f"{base_filename}.wav")
        if not os.path.isfile(wav_path):
            print(f"No matching .wav for '{base_filename}'. Skipping.")
            continue

        print(f"\nGenerating plots for: {base_filename}.wav")
        try:
            y, sr = librosa.load(wav_path, sr=None)
        except Exception as e:
            print(f"Error loading '{wav_path}': {e}")
            continue

        # Known features we might have
        all_features = [
            'mfcc',
            'melspectrogram',
            'spectral_centroid',
            'spectral_flatness',
            'zero',
        ]  # 'zero' covers "zero_crossing_rate" in the file name split

        for ftype in feat_files:
            feature_path = feat_files[ftype]
            # Convert short label 'zero' to actual feature_type
            feature_type = ftype if ftype != 'zero' else 'zero_crossing_rate'
            try:
                feature_data = np.load(feature_path)
                plot_feature(
                    feature_data, y=y, sr=sr, hop_length=512,
                    n_fft=2048, feature_type=feature_type,
                    output_dir=processed_dir, base_filename=base_filename
                )
            except Exception as e:
                print(f"Error plotting '{feature_path}': {e}")

        print(f"Plots saved for '{base_filename}.wav'.")


def main():
    run_plotting()


if __name__ == "__main__":
    main()

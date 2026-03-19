import argparse
import glob
import os
from pathlib import Path
import sys
import warnings

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# Suppress librosa warning about audioread
warnings.filterwarnings("ignore", category=UserWarning)

def plot_feature(feature, y, sr, hop_length,
                 feature_type='mfcc',
                 n_fft=2048,
                 output_dir=None,
                 base_filename=None):
    plt.figure(figsize=(12, 6))
    song_name = base_filename if base_filename else "Unknown Song"

    if feature_type in ['mfcc', 'delta_mfcc', 'delta2_mfcc']:
        librosa.display.specshow(feature, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'{feature_type.upper()} for "{song_name}"')
    
    elif feature_type == 'melspectrogram':
        # Legacy-only path: the active handcrafted extractor does not emit
        # mel-spectrogram arrays by default, but older runs may still have them.
        librosa.display.specshow(librosa.power_to_db(feature, ref=np.max), 
                                 x_axis='time', y_axis='mel',
                                 sr=sr, hop_length=hop_length, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (dB) for "{song_name}"')

    elif feature_type == 'chroma':
        librosa.display.specshow(feature, y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'Chroma Feature for "{song_name}"')

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

    elif feature_type == 'spectral_rolloff':
        times = np.arange(len(y)) / sr
        plt.plot(times, y, alpha=0.4, label='Waveform')
        frames = range(len(feature[0]))
        t_rolloff = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_rolloff, feature[0], color='y', label='Spectral Rolloff')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectral Rolloff for "{song_name}"')
        plt.legend()

    elif feature_type == 'spectral_flux':
        frames = range(len(feature[0]))
        t_flux = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_flux, feature[0], color='c', label='Spectral Flux')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux (Onset Strength)')
        plt.title(f'Spectral Flux for "{song_name}"')
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

    elif feature_type == 'beat_strength':
        # Beat strength features: tempo, mean_onset_strength, std_onset_strength, onset_rate
        # It's a [4, 1] array
        labels = ['Tempo (BPM)', 'Mean Onset', 'Std Onset', 'Onset Rate']
        values = feature.flatten()
        plt.bar(labels, values, color=['skyblue', 'salmon', 'lightgreen', 'plum'])
        plt.title(f'Beat & Rhythm Features for "{song_name}"')
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    else:
        # Generic plot for unknown features if they are 1D/2D
        if feature.ndim == 2 and feature.shape[0] > 1:
            librosa.display.specshow(feature, x_axis='time', sr=sr, hop_length=hop_length)
            plt.title(f'{feature_type.replace("_", " ").title()} for "{song_name}"')
        else:
            plt.plot(feature.flatten())
            plt.title(f'{feature_type.replace("_", " ").title()} for "{song_name}"')

    plt.tight_layout()
    if output_dir and base_filename:
        output_dir = Path(output_dir) / base_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = f"{feature_type}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

def run_plotting(audio_dir='audio_files', features_dir='output/features', plots_dir='output/plots', limit=None):
    audio_dir_path = Path(audio_dir)
    features_dir_path = Path(features_dir)
    plots_dir_path = Path(plots_dir)

    if not features_dir_path.exists():
        print(f"Features directory {features_dir_path} not found. Skipping plotting.")
        return

    plots_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect all audio files (support multiple formats)
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        found = glob.glob(str(audio_dir_path / ext))
        audio_files.extend(found)
    
    if not audio_files:
        # Fallback to genre directories if flat structure doesn't have files
        genre_dirs = [d for d in glob.glob(str(audio_dir_path / "*")) if Path(d).is_dir()]
        for genre_dir in genre_dirs:
            for ext in audio_extensions:
                audio_files.extend(glob.glob(os.path.join(genre_dir, ext)))

    if not audio_files:
        print(f"No audio files found in {audio_dir_path}.")
        return

    if limit:
        audio_files = audio_files[:limit]
        print(f"Limiting to first {limit} songs.")

    print(f"Found {len(audio_files)} files to plot.")

    for wav_path in tqdm(audio_files, desc="Plotting songs"):
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Active handcrafted features extracted by extract_features.py.
        feature_types = [
            'mfcc', 'delta_mfcc', 'delta2_mfcc', 
            'spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 
            'zero_crossing_rate', 'chroma', 'beat_strength'
        ]

        # Only add the legacy mel-spectrogram path if that exact file exists.
        if (features_dir_path / f"{base_filename}_melspectrogram.npy").is_file():
            feature_types.append('melspectrogram')

        try:
            y, sr = librosa.load(wav_path, sr=None)
            
            hop_length = 512 # Default from config
            
            for ft in feature_types:
                feat_path = features_dir_path / f"{base_filename}_{ft}.npy"
                if feat_path.is_file():
                    feature_data = np.load(feat_path)
                    plot_feature(feature_data, y=y, sr=sr, hop_length=hop_length,
                                 feature_type=ft, output_dir=plots_dir_path,
                                 base_filename=base_filename)
            
        except Exception as e:
            print(f"Error plotting features for '{base_filename}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot audio features from extracted data.")
    parser.add_argument(
        "--audio_dir",
        default="audio_files",
        help="Path to the folder containing original audio files (default: audio_files)"
    )
    parser.add_argument(
        "--features_dir",
        default="output/features",
        help="Path to the folder containing feature arrays (default: output/features)"
    )
    parser.add_argument(
        "--plots_dir",
        default="output/plots",
        help="Path to the folder to save plots (default: output/plots)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit the number of songs to plot"
    )
    args = parser.parse_args()
    run_plotting(
        audio_dir=args.audio_dir,
        features_dir=args.features_dir, 
        plots_dir=args.plots_dir,
        limit=args.limit
    )

if __name__ == "__main__":
    main()

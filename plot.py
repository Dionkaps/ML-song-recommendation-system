import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob

def plot_feature(feature, y, sr, hop_length, feature_type='mfcc', n_fft=2048, output_dir=None, base_filename=None):
    """
    Plots a feature (MFCC, Mel-Spectrogram, Spectral Centroid, Spectral Flatness, or Zero-Crossing Rate).
    Saves the plot as a PNG file if output_dir and base_filename are provided.
    
    Parameters:
        feature (np.ndarray): The feature array to plot.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
        hop_length (int): Number of samples between successive frames.
        feature_type (str): Type of feature ('mfcc', 'melspectrogram', 'spectral_centroid', 'spectral_flatness', 'zero_crossing_rate').
        n_fft (int): FFT window size (used for spectral centroid).
        output_dir (str): Directory to save the plot.
        base_filename (str): Base name of the audio file (used in plot title and filename).
    """
    plt.figure(figsize=(10, 4))
    
    # Define the song name for the title
    song_name = base_filename if base_filename else "Unknown Song"
    
    if feature_type == 'mfcc':
        librosa.display.specshow(feature, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'MFCCs for "{song_name}"')
    elif feature_type == 'melspectrogram':
        librosa.display.specshow(feature, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (dB) for "{song_name}"')
    elif feature_type == 'spectral_centroid':
        # Plot waveform
        times = np.arange(len(y)) / sr
        plt.plot(times, y, alpha=0.4, label='Waveform')
        # Overlay spectral centroid
        frames = range(len(feature[0]))
        t_centroid = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        centroid_hz = feature[0]
        plt.plot(t_centroid, centroid_hz, color='r', label='Spectral Centroid')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude / Frequency (Hz)')
        plt.title(f'Spectral Centroid for "{song_name}"')
        plt.legend()
    elif feature_type == 'spectral_flatness':
        # Plot spectral flatness over time
        frames = range(len(feature[0]))
        t_flatness = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_flatness, feature[0], color='g', label='Spectral Flatness')
        plt.xlabel('Time (s)')
        plt.ylabel('Flatness')
        plt.title(f'Spectral Flatness for "{song_name}"')
        plt.legend()
    elif feature_type == 'zero_crossing_rate':
        # Plot zero-crossing rate over time
        frames = range(len(feature[0]))
        t_zcr = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_zcr, feature[0], color='m', label='Zero-Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Rate')
        plt.title(f'Zero-Crossing Rate for "{song_name}"')
        plt.legend()
    else:
        raise ValueError("feature_type must be 'mfcc', 'melspectrogram', 'spectral_centroid', 'spectral_flatness', or 'zero_crossing_rate'")
    
    plt.tight_layout()
    
    if output_dir and base_filename:
        plot_filename = f"{base_filename}_{feature_type}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    else:
        plt.show()

def main():
    # Define the input directory and results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio_files")
    results_dir = os.path.join(current_dir, "results")
    
    # Ensure the results directory exists
    if not os.path.isdir(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        return
    
    # Find all .wav files in the audio_files directory
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print("No .wav files found in the audio_files directory.")
        return
    
    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing plots for: {base_filename}.wav")
        
        # Define paths to feature files
        feature_files = {
            'mfcc': os.path.join(results_dir, f"{base_filename}_mfcc.npy"),
            'melspectrogram': os.path.join(results_dir, f"{base_filename}_melspectrogram.npy"),
            'spectral_centroid': os.path.join(results_dir, f"{base_filename}_spectral_centroid.npy"),
            'spectral_flatness': os.path.join(results_dir, f"{base_filename}_spectral_flatness.npy"),
            'zero_crossing_rate': os.path.join(results_dir, f"{base_filename}_zero_crossing_rate.npy")
        }
        
        # Check if all feature files exist
        missing_features = [ft for ft, path in feature_files.items() if not os.path.isfile(path)]
        if missing_features:
            print(f"Missing feature files for '{base_filename}.wav': {', '.join(missing_features)}. Skipping.")
            continue
        
        try:
            # Load audio to get y and sr
            y, sr = librosa.load(wav_path, sr=None)
            print(f"Loaded '{base_filename}.wav' with sampling rate = {sr} Hz")
            
            # Load features
            mfccs = np.load(feature_files['mfcc'])
            mel_spectrogram = np.load(feature_files['melspectrogram'])
            spectral_centroid = np.load(feature_files['spectral_centroid'])
            spectral_flatness = np.load(feature_files['spectral_flatness'])
            zero_crossing_rate = np.load(feature_files['zero_crossing_rate'])
            
            # Define extraction parameters (should match feature extraction script)
            hop_length = 512
            n_fft = 2048
            
            # Plot and save each feature with song name in the title
            plot_feature(
                feature=mfccs, 
                y=y, 
                sr=sr, 
                hop_length=hop_length, 
                feature_type='mfcc', 
                output_dir=results_dir, 
                base_filename=base_filename
            )
            plot_feature(
                feature=mel_spectrogram, 
                y=y, 
                sr=sr, 
                hop_length=hop_length, 
                feature_type='melspectrogram', 
                output_dir=results_dir, 
                base_filename=base_filename
            )
            plot_feature(
                feature=spectral_centroid, 
                y=y, 
                sr=sr, 
                hop_length=hop_length, 
                feature_type='spectral_centroid', 
                n_fft=n_fft, 
                output_dir=results_dir, 
                base_filename=base_filename
            )
            plot_feature(
                feature=spectral_flatness, 
                y=y, 
                sr=sr, 
                hop_length=hop_length, 
                feature_type='spectral_flatness', 
                output_dir=results_dir, 
                base_filename=base_filename
            )
            plot_feature(
                feature=zero_crossing_rate, 
                y=y, 
                sr=sr, 
                hop_length=hop_length, 
                feature_type='zero_crossing_rate', 
                output_dir=results_dir, 
                base_filename=base_filename
            )
            
            print(f"Plots for '{base_filename}.wav' saved successfully.")
        
        except Exception as e:
            print(f"An error occurred while plotting features for '{base_filename}.wav': {e}")

if __name__ == "__main__":
    main()
